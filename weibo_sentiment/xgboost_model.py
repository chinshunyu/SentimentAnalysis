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
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from utils import load_corpus, stopwords

TRAIN_PATH = "./train.txt"
TEST_PATH = "./test.txt"

train_data = load_corpus(TRAIN_PATH)
test_data = load_corpus(TEST_PATH)

df_train = pd.DataFrame(train_data, columns=["words", "label"])
df_test = pd.DataFrame(test_data, columns=["words", "label"])


vectorizer = CountVectorizer(
    token_pattern="\[?\w+\]?", stop_words=stopwords, max_features=2000
)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]

X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]

param = {
    "booster": "gbtree",
    "max_depth": 6,
    "scale_pos_weight": 0.5,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "error",
    "eta": 0.3,
    "nthread": 10,
}

dmatrix = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(param, dmatrix, num_boost_round=200)

dmatrix = xgb.DMatrix(X_test)
y_pred = model.predict(dmatrix)


auc_score = metrics.roc_auc_score(y_test, y_pred)  # 先计算AUC
y_pred = list(map(lambda x: 1 if x > 0.5 else 0, y_pred))  # 二值化
print(metrics.classification_report(y_test, y_pred))
print("准确率:", metrics.accuracy_score(y_test, y_pred))
print("AUC:", auc_score)


# 保存模型
model.save_model("xgboost_model.model")

# 加载模型
model = xgb.Booster()
model.load_model("xgboost_model.model")


# 预测
from utils import processing

strs = ["哈哈哈哈哈笑死我了", "我也是有脾气的!"]
words = [processing(s) for s in strs]
vec = vectorizer.transform(words)
dmatrix = xgb.DMatrix(vec)
output = model.predict(dmatrix)
print(output)
