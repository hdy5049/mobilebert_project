import torch
import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_linear_schedule_with_warmup, logging
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

#  디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
logging.set_verbosity_error()

#  데이터 로딩 및 전처리
path = "C:/Users/108-0/PycharmProjects/mobilebert_project/gta5_comments_sentiment_only.csv"
df = pd.read_csv(path)

if 'Comments' in df.columns:
    df = df.rename(columns={'Comments': 'Text'})
elif 'User Review Text' in df.columns:
    df = df.rename(columns={'User Review Text': 'Text'})

df['Text'] = df['Text'].astype(str).str.strip()
df['Text'] = df['Text'].apply(lambda x: re.sub(r'\s+', ' ', x))

def is_probably_english(text):
    if not text or len(text) < 5:
        return False
    english_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
    return english_ratio > 0.6

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df = df[df['Text'].apply(is_probably_english)]
df['Text'] = df['Text'].apply(clean_text)
df = df[df['Text'].str.len() >= 5]

if 'sentiment' in df.columns:
    df = df.rename(columns={'sentiment': 'Sentiment'})
elif 'User Rating' in df.columns:
    df['score'] = pd.to_numeric(df['User Rating'], errors='coerce')
    def label_by_score(score):
        if score >= 35:
            return 1
        elif score <= 30:
            return 0
    df['Sentiment'] = df['score'].apply(label_by_score)

df = df[df['Sentiment'].notnull()]
df = df[['Text', 'Sentiment']]

#  총 데이터 수 출력
print(f"총 전체 데이터 수: {len(df)}")

#  train/test 완전 분리 및 학습데이터 축소+노이즈 추가
X = df['Text'].tolist()
y = df['Sentiment'].astype(int).tolist()
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  학습 데이터 줄이기 더 강하게 (20%)
train_subset_size = int(len(X_trainval) * 0.2)
X_trainval = X_trainval[:train_subset_size]
y_trainval = y_trainval[:train_subset_size]

#  라벨 오염 비율 증가 (15%)
num_noise = int(0.15 * len(y_trainval))
noise_indices = random.sample(range(len(y_trainval)), num_noise)
for idx in noise_indices:
    y_trainval[idx] = 1 - y_trainval[idx]  # 0↔1 뒤집기

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

#  토큰화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_enc = tokenizer(X_train, truncation=True, max_length=256, padding="max_length", add_special_tokens=True)
val_enc   = tokenizer(X_val, truncation=True, max_length=256, padding="max_length", add_special_tokens=True)
test_enc  = tokenizer(X_test, truncation=True, max_length=256, padding="max_length", add_special_tokens=True)

train_inputs = torch.tensor(train_enc['input_ids'], dtype=torch.long)
train_masks  = torch.tensor(train_enc['attention_mask'], dtype=torch.long)
train_labels = torch.tensor(y_train, dtype=torch.long)

val_inputs   = torch.tensor(val_enc['input_ids'], dtype=torch.long)
val_masks    = torch.tensor(val_enc['attention_mask'], dtype=torch.long)
val_labels   = torch.tensor(y_val, dtype=torch.long)

test_inputs  = torch.tensor(test_enc['input_ids'], dtype=torch.long)
test_masks   = torch.tensor(test_enc['attention_mask'], dtype=torch.long)
test_labels  = torch.tensor(y_test, dtype=torch.long)

#  DataLoader 구성
batch_size = 8
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)

#  클래스 가중치 적용
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

#  모델 설정
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)*epochs)

#  평가 함수
def evaluate(dataloader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids, mask, labels = [x.to(device) for x in batch]
            outputs = model(ids, attention_mask=mask)
            pred = torch.argmax(outputs.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
    return np.mean(np.array(preds) == np.array(truths))

#  학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        ids, mask, labels = [x.to(device) for x in batch]
        labels = labels.long()
        model.zero_grad()
        outputs = model(ids, attention_mask=mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        loop.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    print(f"\n📘 Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

#  테스트셋으로 최종 평가
model.eval()
preds, truths = [], []
for batch in tqdm(test_loader, desc=" 테스트셋 예측"):
    ids, mask, labels = [x.to(device) for x in batch]
    with torch.no_grad():
        outputs = model(ids, attention_mask=mask)
    pred = torch.argmax(outputs.logits, dim=1)
    preds.extend(pred.cpu().numpy())
    truths.extend(labels.cpu().numpy())

final_acc = np.mean(np.array(preds) == np.array(truths))
print(f"\n🎯 최종 테스트셋 정확도: {final_acc * 100:.2f}%")
