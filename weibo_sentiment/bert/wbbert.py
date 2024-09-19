# 预训练模型下载：https://github.com/ymcui/Chinese-BERT-wwm
import os

import pandas as pd
import pretty_errors
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from utils import load_corpus_bert, processing_bert

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


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_corpus: list[tuple[str, int]] = load_corpus_bert("./train.txt")
test_corpus: list[tuple[str, int]] = load_corpus_bert("./test.txt")
df_train: pd.DataFrame = pd.DataFrame(train_corpus, columns=["text", "label"])
df_test: pd.DataFrame = pd.DataFrame(test_corpus, columns=["text", "label"])
tokenizer = BertTokenizer.from_pretrained("./chinese_wwm_pytorch")
bert_model = BertModel.from_pretrained("./chinese_wwm_pytorch")
bert_model.to(device)


class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super(MyDataset, self).__init__()
        self.text: list[str] = df["text"].tolist()
        self.label: list[int] = df["label"].tolist()
        assert len(self.text) == len(self.label)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.text[index], self.label[index]


train_dataset: MyDataset = MyDataset(df_train)
test_dataset: MyDataset = MyDataset(df_test)
train_loader: DataLoader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader: DataLoader = DataLoader(test_dataset, batch_size=100, shuffle=False)


class Net(nn.Module):
    def __init__(self, input_size) -> None:
        super(Net, self).__init__()
        self.fc: nn.Module = nn.Linear(input_size, 1)
        self.sigmoid: nn.Module = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc(x)
        x = self.sigmoid(x)
        return x


def test(net: Net, test_loader: DataLoader, device: torch.device) -> float:
    net.to(device)
    net.eval()
    correct: int = 0
    total: int = 0
    y_pred: list[int] = []
    y_true: list[int] = []
    with torch.no_grad():
        for i, mini_batch_100 in enumerate(test_loader):
            # print(mini_batch_100[0])
            sents: tuple[str] = mini_batch_100[0]
            # print(type(sents))
            # import sys
            # sys.exit()
            labels: torch.Tensor[float] = mini_batch_100[1].float().to(device)
            tokens = tokenizer(
                sents, return_tensors="pt", padding=True, truncation=True
            )
            input_ids: torch.Tensor[int] = tokens["input_ids"].to(device)  # 100个句子
            attention_mask: torch.Tensor[int] = tokens["attention_mask"].to(device)
            bert_outputs: torch.Tensor[int] = bert_model(
                input_ids, attention_mask=attention_mask
            )[1]
            outputs: torch.Tensor[float] = net(bert_outputs).view(-1)
            y_pred.append(outputs)
            y_true.append(labels)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    print(metrics.classification_report(y_true.cpu().numpy(), y_pred.cpu().numpy()))
    print("acc:", metrics.accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))
    print("AUC:", metrics.roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))


def train(
    net: Net,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    net.to(device)
    criterion: nn.Module = nn.BCELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    )
    for epoch in range(num_epochs):
        total_loss: float = 0.0
        net.train()
        for i, mini_batch_100 in enumerate(train_loader):
            sents: tuple[str] = mini_batch_100[0]
            labels: torch.Tensor[float] = mini_batch_100[1].float().to(device)
            tokens = tokenizer(
                sents, padding=True, truncation=True, return_tensors="pt"
            )
            input_ids: torch.Tensor[int] = tokens["input_ids"].to(device)  # 100个句子
            attention_mask: torch.Tensor[int] = tokens["attention_mask"].to(device)
            with torch.no_grad():
                bert_outputs: torch.Tensor[int] = bert_model(
                    input_ids, attention_mask=attention_mask
                )[1]
            outputs: torch.Tensor[float] = net(bert_outputs)
            logits: torch.Tensor[float] = outputs.view(-1)
            loss: torch.Tensor[float] = criterion(logits, labels)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, iter {i+1}, Loss:{total_loss / 10}")
                total_loss: float = 0.0
        lr_scheduler.step()
        test(net, test_loader, device)

        model_path: str = f"./bert_finetuned_model_{epoch}.pth"
        torch.save(net, model_path)
        print(f"saved model:{model_path}")


# def use_model():
#     net = torch.load("./bert_finetuned_model_8.pth")
#     s = ["华丽繁荣的城市、充满回忆的小镇、郁郁葱葱的山谷...", "突然就觉得人间不值得"]
#     tokens = tokenizer(s, padding=True)
#     input_ids = torch.tensor(tokens["input_ids"])
#     attention_mask = torch.tensor(tokens["attention_mask"])
#     last_hidden_states = bert_model(input_ids, attention_mask=attention_mask)
#     bert_output = last_hidden_states[0][:, 0]
#     outputs = net(bert_output)
#     print(outputs)

if __name__ == "__main__":
    bert_output_size: int = bert_model.config.hidden_size
    net: Net = Net(bert_output_size)
    num_epochs: int = 10
    lr: float = 1e-3
    train(net, train_loader, test_loader, num_epochs, lr, device)
    # test(net, test_loader, device)
