import pandas as pd
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_corpus, stopwords

TRAIN_PATH = "./train.txt"
TEST_PATH = "./test.txt"

train_data = load_corpus(TRAIN_PATH)
test_data = load_corpus(TEST_PATH)

df_train = pd.DataFrame(train_data, columns=["words", "label"])
df_test = pd.DataFrame(test_data, columns=["words", "label"])


vectorizer = TfidfVectorizer(token_pattern="\[?\w+\]?", stop_words=stopwords)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]

X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]

clf = svm.SVC()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("准确率:", metrics.accuracy_score(y_test, y_pred))

# 预测
import joblib

# 保存模型
model_filename = "svm_model.pkl"
joblib.dump(clf, model_filename)

# 保存向量化器
vectorizer_filename = "tfidf_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_filename)


clf = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

from utils import processing

strs = ["只要流过的汗与泪都能化作往后的明亮，就值得你为自己喝彩", "烦死了！为什么周末还要加班[愤怒]"]
words = [processing(s) for s in strs]
vec = vectorizer.transform(words)
output = clf.predict(vec)

print(output)
