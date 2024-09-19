import numpy as np
import torch
from bert.wbbert import Net, bert_model, device, tokenizer

bert_model.to(device)
net = torch.load("./model_9.pth")
sentences = ["好烦啊", "今天真开心", "今天是9月11日星期三"]


def inference(sentences: list[str]) -> np.ndarray:
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]
    logits = net(bert_output)
    print(logits)
    predictions = (logits >= 0.5).long().view(-1).cpu().numpy()
    print(predictions)
    return predictions


if __name__ == "__main__":
    inference(sentences)
