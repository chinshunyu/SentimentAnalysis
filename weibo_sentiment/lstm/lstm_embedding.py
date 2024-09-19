import os

import pretty_errors

os.system("clear")
pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=5,
    lines_after=2,
    line_color=pretty_errors.RED + "> " + pretty_errors.BRIGHT_RED,
    filename_color=pretty_errors.YELLOW,
    header_color=pretty_errors.BRIGHT_GREEN,
    link_color=pretty_errors.BRIGHT_BLUE,
    code_color=pretty_errors.BRIGHT_RED,
    # code_color='  '+pretty_errors.default_config.line_color
    line_number_color=pretty_errors.BRIGHT_RED,
)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import sys
from os.path import join as opj

from utils import load_corpus

TRAIN_PATH = "./train.txt"
TEST_PATH = "./test.txt"
train_data = load_corpus(TRAIN_PATH)
test_data = load_corpus(TEST_PATH)
import pandas as pd

df_train = pd.DataFrame(train_data, columns=["text", "label"])
df_test = pd.DataFrame(test_data, columns=["text", "label"])

from collections import Counter

import torch
from sklearn import metrics
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

learning_rate = 5e-4
# input_size = 768
num_epoches = 5
batch_size = 100
embed_size = 64
hidden_size = 64
num_layers = 2


def build_vocab(texts: list[str]):
    word_counter = Counter()
    for text in texts:
        word_counter.update(text.split(" "))
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counter.most_common())}
    vocab["<UNK>"] = 0
    return vocab


vocab = build_vocab(df_train["text"].tolist() + df_test["text"].tolist())
vocab_size = len(vocab)


class MyDataset(Dataset):
    def __init__(self, df, vocab):
        self.data: list[torch.Tensor] = []  # 以句子每单位，所有句子的词序号列表
        self.label = df["label"].tolist()
        self.vocab = vocab
        for sent in df["text"].tolist():
            # 获取一个句子中每个单词在此表中对应的序号
            indices: list[int] = [
                self.vocab.get(word, self.vocab["<UNK>"]) for word in sent.split(" ")
            ]
            self.data.append(torch.tensor(indices, dtype=torch.long))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self) -> int:
        return len(self.label)


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    batch_length: list[int] = [len(sq[0]) for sq in batch]
    x = [i[0] for i in batch]
    y = [i[1] for i in batch]
    batch = pad_sequence(x, batch_first=True, padding_value=0)
    return batch, torch.tensor(y, dtype=torch.long), batch_length


train_dataset = MyDataset(df_train, vocab)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_dataset = MyDataset(df_test, vocab)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向, 输出维度要*2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        x = self.embedding(x)  # 将输入的词索引转换为词向量
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            device
        )  # 双向, 第一个维度要*2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            input=x, lengths=lengths, batch_first=True
        )
        packed_out, (h_n, h_c) = self.lstm(packed_input, (h0, c0))
        lstm_out = torch.cat(
            [h_n[-2], h_n[-1]], 1
        )  # 双向, 所以要将最后两维拼接, 得到的就是最后一个time step的输出
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out


lstm = LSTM(vocab_size, embed_size, hidden_size, num_layers).to(device)


# 在测试集效果检验
def test():
    y_pred, y_true = [], []

    with torch.no_grad():
        for x, labels, lengths in test_dataloader:
            x = x.to(device)
            outputs = lstm(x, lengths)  # 前向传播
            outputs = outputs.view(-1)  # 将输出展平
            y_pred.append(outputs)
            y_true.append(labels)

    y_prob = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    y_pred = y_prob.clone()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    print(metrics.classification_report(y_true, y_pred.cpu().numpy()))
    print("acc:", metrics.accuracy_score(y_true, y_pred.cpu().numpy()))
    print("AUC:", metrics.roc_auc_score(y_true, y_prob.cpu().numpy()))


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)


def train():
    for epoch in range(num_epoches):
        total_loss = 0
        for i, (x, labels, lengths) in enumerate(train_dataloader):
            x = x.to(device)
            labels = labels.to(device).float()
            outputs = lstm(x, lengths)  # 前向传播
            logits = outputs.view(-1)  # 将输出展平
            loss = criterion(logits, labels)  # loss计算
            total_loss += loss
            optimizer.zero_grad()  # 梯度清零
            loss.backward(retain_graph=True)  # 反向传播，计算梯度
            optimizer.step()  # 梯度更新
            if (i + 1) % 10 == 0:
                print(
                    "epoch:{}, step:{}, loss:{}".format(
                        epoch + 1, i + 1, total_loss / 10
                    )
                )
                total_loss = 0

        # test
        test()

        # save model
        model_path = "./lstm_embed_model{}.pth".format(epoch + 1)
        torch.save(lstm, model_path)
        print("saved model: ", model_path)


def inference(input_texts: list[str]):
    from utils import processing

    lstm = torch.load(opj(PROJECT_ROOT, "lstm_embed_model5.pth"))
    lstm.eval()
    # 处理输入文本

    def text_to_indices(texts, vocab):
        indices_list = []
        for text in texts:
            words = processing(text).split(" ")
            indices = [vocab.get(word, vocab["<UNK>"]) for word in words]
            indices_list.append(torch.tensor(indices, dtype=torch.long))
        return indices_list

    input_indices_list = text_to_indices(input_texts, vocab)
    # 使用 collate_fn 处理批量数据
    batch = [(indices, 0) for indices in input_indices_list]  # 标签可以任意设置，因为我们不使用标签
    batch, _, lengths = collate_fn(batch)
    batch = batch.to(device)

    # 进行推理
    with torch.no_grad():
        outputs = lstm(batch, lengths)
    # 处理输出
    outputs = outputs.view(-1).cpu().numpy()
    predicted_classes = [1 if output > 0.5 else 0 for output in outputs]
    for i, text in enumerate(input_texts):
        print(f"Input Text: {text}")
        print(f"Predicted Class: {predicted_classes[i]}")
        print(f"Probability: {outputs[i]}")


if __name__ == "__main__":
    train()
    input_texts = [
        "总觉得隔壁桌点的比较好吃",
        "嗯我喜欢随大流生活，活出自己人生节奏什么的太焦虑了[汗]怎么办一点也不酷，人格分裂到每天脑子里有个姑婆不停叨叨成家立业年纪不等人的同时玩耍自嗨向往远方",
        "三亚好吃又便宜的芒果这么多，贵妃芒，苹果芒，鸡蛋芒，尖椒芒，台芒……我为什么要买这个又贵又难吃的奥芒",
    ]
    inference(input_texts)
