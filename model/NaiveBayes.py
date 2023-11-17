from sklearn.naive_bayes import MultinomialNB
import joblib

class NaiveBayes():
    def __init__(self, alpha=10, fit_prior=True):
        self.model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    
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