import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

#  1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#  2. 리뷰 데이터 로드
data_path = "C:/Users/108-0/PycharmProjects/mobilebert_project/gta5_comments_3000.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig")

#  3. 텍스트와 라벨 추출
data_X = df['Comments'].astype(str).tolist()
labels = df['sentiment'].astype(int).tolist()
print("총 리뷰 수:", len(data_X))

#  4. Tokenizer
tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased", do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", add_special_tokens=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("토큰화 완료")

#  5. DataLoader 구성
batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)
test_dataset = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, sampler=torch.utils.data.SequentialSampler(test_dataset),
                                          batch_size=batch_size)
print("데이터 구축 완료")

#  6. 학습된 모델 로드
model = MobileBertForSequenceClassification.from_pretrained("mobilebert-uncased", num_labels=2)
model.load_state_dict(torch.load("C:/Users/108-0/PycharmProjects/mobilebert_project/gta5_mobilebert_finetuned.pt"))
model.to(device)
model.eval()

#  7. 추론
test_pred = []
test_true = []
progress = tqdm(total=len(test_dataset), desc="리뷰 기준 추론 중")

for batch in test_loader:
    input_ids, attention_masks, labels_batch = [x.to(device) for x in batch]
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)

    test_pred.extend(preds.cpu().numpy())
    test_true.extend(labels_batch.cpu().numpy())
    progress.update(len(batch[0]))

progress.close()

#  8. 정확도 출력
accuracy = np.mean(np.array(test_pred) == np.array(test_true))
print(f"\n🎯 GTA5 리뷰 긍부정 정확도: {accuracy * 100:.2f}%")
