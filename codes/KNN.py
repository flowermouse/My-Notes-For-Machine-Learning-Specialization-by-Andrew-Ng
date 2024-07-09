import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # 计算距离
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # 获取最近的k个元素的索引
        k_indices = np.argsort(distances)[:self.k]
        # 提取这k个元素的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 返回最常见的标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]