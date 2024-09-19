import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import load_corpus, processing, stopwords

TRAIN_PATH = "./train.txt"
TEST_PATH = "./test.txt"

train_data = load_corpus(TRAIN_PATH)
test_data = load_corpus(TEST_PATH)

import pandas as pd

df_train = pd.DataFrame(train_data, columns=["words", "label"])
df_test = pd.DataFrame(test_data, columns=["words", "label"])

vectorizer = CountVectorizer(token_pattern="\[?\w+\]?", stop_words=stopwords)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]

X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print(metrics.classification_report(y_test, y_pred))
print("准确率:", metrics.accuracy_score(y_test, y_pred))


import joblib

# 保存模型
joblib.dump(clf, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# 加载模型
clf = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# 预测
strs = ["终于收获一个最好消息", "哭了, 今天怎么这么倒霉"]
words = [processing(s) for s in strs]
vec = vectorizer.transform(words)

output = clf.predict(vec)
print(output)
