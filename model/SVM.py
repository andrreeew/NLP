from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

class SVM():
    def __init__(self):
        self.model = SVC(kernel='linear', C=1.0)
        self.scaler = StandardScaler()

    def train(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
    
    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))
    
    def save(self, path):
        joblib.dump(self.model, path)
        print('保存到:', path)


    def load(self, path):
        self.model = joblib.load(path)
        print('加载模型:', path)