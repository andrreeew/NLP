from sklearn.tree import DecisionTreeClassifier
import joblib

class DecisionTree():
    def __init__(self, max_depth=7, min_samples_split=10):
        self.model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
        print('保存到:', path)


    def load(self, path):
        self.model = joblib.load(path)
        print('加载模型:', path)