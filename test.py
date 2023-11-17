from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


device = torch.device("cpu")


model_name = 'nlp/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)
    
    #  text = self.texts[idx]
        # label = self.labels[idx]
        # encoding = tokenizer(text, truncation=True, padding=True, return_tensors='pt', max_length=128)
        # return {'input_ids': encoding['input_ids'].flatten(),
        #         'attention_mask': encoding['attention_mask'].flatten(),
        #         'labels': torch.tensor(label, dtype=torch.long)}

    def __getitem__(self, idx):
        # encoding = self.tokenizer(, return_tensors='pt', padding=True, truncation=True, max_length=200)
        
        encoding = self.tokenizer(self.texts[idx],padding = 'max_length',truncation = True,max_length = 128,return_tensors='pt')  
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx])
        }
        return item
    
    
def test(model, testloader):
    model.eval()
    y = []
    y_pred = []
    for batch in tqdm(iter(testloader)):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        output = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
        predicted_label = torch.argmax(output.logits, dim=1).tolist()
        
        y += labels.tolist()
        y_pred += predicted_label
    
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y, y_pred))

        
        
    

def train(model, dataloader, testloader, num_epochs=10, lr=1e-5):    
    model.train()  
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for batch in tqdm(iter(dataloader)):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('epoch {}'.format(epoch+1))
        test(model, testloader)
        torch.save(model.state_dict(), 'bert_sentiment_model.pth')
            
        

    


import pandas as pd

pd_all = pd.read_csv('nlp/weibo_senti_100k.csv')
moods = {0: '负向', 1: '正向'}

# pd_all = pd.read_csv('nlp/simplifyweibo_4_moods.csv')
# moods = {0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}

print('微博数目（总体）：%d' % pd_all.shape[0])

for label, mood in moods.items(): 
    print('微博数目（{}): {}'.format(mood,  pd_all[pd_all.label==label].shape[0]))

s = pd_all.sample(10000)
texts = [item[1] for item in s.values]
labels = [item[0] for item in s.values]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


dataset = CustomDataset(X_train, y_train, tokenizer)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


test_dataset = CustomDataset(X_test, y_test, tokenizer)
testloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

train(model, dataloader, testloader)
