import paddlehub as hub

senta = hub.Module(name="senta_bilstm")
test_text = ["天啦，千万别多说，扰乱军心，哈哈", "该做什么的时候就得好好做，别多想了"]


def senti_classify(input_text):
    input_dict = {"text": input_text}
    results = senta.sentiment_classify(data=input_dict)

    return results


if __name__ == "__namin__":
    print(senti_classify(test_text))
