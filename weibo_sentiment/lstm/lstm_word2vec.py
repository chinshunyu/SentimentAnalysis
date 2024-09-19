import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from os.path import join as opj
from typing import Any

import numpy as np
import pandas as pd
import torch
from gensim import models
from sklearn import metrics
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from utils import load_corpus

train_data: list[tuple[str, int]] = load_corpus(opj(PROJECT_ROOT, "train.txt"))
test_adta: list[tuple[str, int]] = load_corpus(opj(PROJECT_ROOT, "test.txt"))
df_train: pd.DataFrame = pd.DataFrame(train_data, columns=["text", "label"])
df_test: pd.DataFrame = pd.DataFrame(test_adta, columns=["text", "label"])


def train_w2v_mdoel():
    w2v_input = (df_train["text"]).map(lambda x: x.split(" "))
    w2v_model = models.Word2Vec(
        w2v_input, vector_size=64, window=5, min_count=1, workers=4, epochs=1000
    )
    w2v_model.save("./w2v.model")


class MyDataSet(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data: list[torch.Tensor] = []
        self.label: list[int] = df["label"].tolist()
        for sent in df["text"].to_list():
            sent_vectors: list[np.ndarray] = []
            for word in sent.split(" "):
                if word in w2v_model.wv.key_to_index:
                    sent_vectors.append(w2v_model.wv[word])
            sent_vectors: torch.Tensor = torch.tensor(np.array(sent_vectors))
            self.data.append(sent_vectors)

    def __getitem__(self, index) -> Any:
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    data_length: list[int] = [len(item[0]) for item in batch]
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    data = pad_sequence(x, batch_first=True, padding_value=0)
    return data, torch.tensor(y, dtype=torch.float32), data_length


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        packed_out, (hn, hc) = self.lstm(packed_input, (h0, c0))
        lstm_out = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out


def test(test_dataloader):
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, labels, lengths in test_dataloader:
            x = x.to(device)
            outputs = lstm(x, lengths)
            outputs = outputs.view(-1)
            y_pred.append(outputs)
            y_true.append(labels)
    y_prob = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    y_pred = y_prob.clone()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_pred_cpu = y_pred.cpu().numpy()
    print(metrics.classification_report(y_true, y_pred_cpu))
    print(f"acc:{metrics.accuracy_score(y_pred=y_pred_cpu, y_true=y_true)}")
    print(f"AUC:{metrics.roc_auc_score(y_true, y_prob.cpu().numpy())}")


def train(train_dataloader, test_dataloader):
    for epoch in range(num_epoches):
        total_loss = 0
        for i, (x, labels, lengths) in enumerate(train_dataloader):
            x = x.to(device)
            labels = labels.to(device)
            outputs = lstm(x, lengths)
            logits = outputs.view(-1)
            loss = creterion(logits, labels)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    "epoch:{}, step:{}, loss:{}".format(
                        epoch + 1, i + 1, total_loss / 10
                    )
                )
                total_loss = 0
        test(test_dataloader)

        model_path = "./lstm_model_{}.pth".format(epoch + 1)
        torch.save(lstm, model_path)
        print("saved model: ", model_path)


def inference(input_texts: list[str]):
    from utils import processing

    lstm = torch.load("./lstm_model_5.pth")
    data = []
    for s in input_texts:
        vectors = []
        for w in processing(s).split(" "):
            if w in w2v_model.wv.key_to_index:
                vectors.append(w2v_model.wv[w])  # 将每个词替换为对应的词向量
        vectors = torch.Tensor(vectors)
        data.append(vectors)
    x, _, lengths = collate_fn(list(zip(data, [-1] * len(input_texts))))
    with torch.no_grad():
        x = x.to(device)
        outputs = lstm(x, lengths)  # 前向传播
        outputs = outputs.view(-1)  # 将输出展平
    print(outputs)


if __name__ == "__main__":
    train_w2v_mdoel()
    w2v_model = models.Word2Vec.load("./w2v.model")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    learning_rate = 5e-4
    # input_size = 768
    num_epoches = 5
    batch_size = 100
    embed_size = 64
    hidden_size = 64
    num_layers = 2

    lstm = LSTM(embed_size, hidden_size, num_layers).to(device)
    creterion = nn.BCELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # 训练集
    train_data = MyDataSet(df_train)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    # 测试集
    test_data = MyDataSet(df_test)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    train(train_dataloader, test_dataloader)
    input_texts = [
        "总觉得隔壁桌点的比较好吃",
        "嗯我喜欢随大流生活，活出自己人生节奏什么的太焦虑了[汗]怎么办一点也不酷，人格分裂到每天脑子里有个姑婆不停叨叨成家立业年纪不等人的同时玩耍自嗨向往远方",
        "三亚好吃又便宜的芒果这么多，贵妃芒，苹果芒，鸡蛋芒，尖椒芒，台芒……我为什么要买这个又贵又难吃的奥芒",
    ]
    inference(input_texts)
