from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
import torch



class BERT():
    # def __init__(self):
    #     model_name = './bert-base-chinese'
    #     self.tokenizer = BertTokenizer.from_pretrained(model_name)
    #     self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    
    def predict(self, X):
        y = torch.argmax(self.model(torch.tensor(X).float()), -1)
        encoding = self.tokenizer(X, padding = 'max_length',truncation = True,max_length = 128,return_tensors='pt')
        output = self.model(**encoding)
        y = torch.argmax(output.logits, dim=1).item()
        return y
    
    def save(self, path):
        self.model.save_pretrained(path)
        print('保存到:', path)
        

    def load(self, path):
        # self.model = BertForSequenceClassification.from_pretrained(path, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path, num_labels=2)
        print('加载模型:', path)