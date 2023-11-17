from model.DecisionTree import DecisionTree
from model.DNN import DNN
from model.NaiveBayes import NaiveBayes
from model.SVM import SVM
import numpy as np
import jieba
from transformers import BertTokenizer, BertForSequenceClassification
import torch


def get_text_encoding(texts, dictionary):
    # texts: [text1, text2, ...]
    # dictionary: [word1, word2, ....]

    encoding_size = len(dictionary)
    text_encoding = np.zeros((len(texts), encoding_size))
    for textId in range(len(texts)):
        words = set(jieba.cut(texts[textId]))
        for word in words:
            for dicId  in range(encoding_size):
                if word == dictionary[dicId]:
                    text_encoding[textId, dicId] = 1
                    break
            
    return text_encoding


dictionary = []
with open('dictionary.txt', 'r') as file:
    for line in file:
        dictionary.append(line.strip())
print(dictionary)


decision_tree = DecisionTree()
naive_bayes = NaiveBayes()
dnn = DNN()

decision_tree.load('param/decision_tree_model.pth')
naive_bayes.load('param/naive_bayes_model.pth')
dnn.load('param/dnn_model.pth')

model_name = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
model = BertForSequenceClassification.from_pretrained('./bert-param')


def predict(text):
  
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    output = model(**encoding)
    predicted_label = torch.argmax(output.logits, dim=1).item()

    text_encoding = get_text_encoding([text], dictionary)
    return [decision_tree.predict(text_encoding)[0], 
            naive_bayes.predict(text_encoding)[0], 
            dnn.predict(text_encoding)[0], predicted_label]

print(predict('测试'))

