import time
import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold



class ImprovedTree:
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.prediction = None

    def fit(self, X, y):
        if len(np.unique(y)) == 1:
            self.prediction = y[0]
            return
        best_split_score = -np.inf
        for feature in range(X.shape[1]):
            for split_value in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= split_value
                right_mask = X[:, feature] > split_value
                left_score = self.score(y[left_mask])
                right_score = self.score(y[right_mask])
                split_score = left_score + right_score
                if split_score > best_split_score:
                    best_split_score = split_score
                    self.split_feature = feature
                    self.split_value = split_value
        left_mask = X[:, self.split_feature] <= self.split_value
        right_mask = X[:, self.split_feature] > self.split_value
        self.left = ImprovedTree()
        self.right = ImprovedTree()
        self.left.fit(X[left_mask], y[left_mask])
        self.right.fit(X[right_mask], y[right_mask])

    def score(self, y):
        mean = np.mean(y)
        return -np.sum((y - mean) ** 2)

    def predict(self, X):
        if self.prediction is not None:
            return np.full(X.shape[0], self.prediction)
        left_mask = X[:, self.split_feature] <= self.split_value
        right_mask = X[:, self.split_feature] > self.split_value
        predictions = np.zeros(X.shape[0])
        predictions[left_mask] = self.left.predict(X[left_mask])
        predictions[right_mask] = self.right.predict(X[right_mask])
        return predictions

class SimpleXGBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        residuals = y.astype(float)  # Ensure residuals are float for calculations
        for _ in range(self.n_estimators):
            tree = ImprovedTree()
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return np.round(predictions)

    def save_model(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{self.n_estimators},{self.learning_rate}\n")
            for tree in self.trees:
                self._save_tree(tree, file)

    def _save_tree(self, tree, file):
        if tree.prediction is not None:
            file.write(f"leaf,{tree.prediction}\n")
        else:
            file.write(f"node,{tree.split_feature},{tree.split_value}\n")
            self._save_tree(tree.left, file)
            self._save_tree(tree.right, file)

    def load_model(self, filename):
        with open(filename, 'r') as file:
            line = file.readline().strip()
            parts = line.split(',')
            self.n_estimators = int(parts[0])
            self.learning_rate = float(parts[1])
            self.trees = []
            for _ in range(self.n_estimators):
                tree = ImprovedTree()
                self._load_tree(tree, file)
                self.trees.append(tree)

    def _load_tree(self, tree, file):
        line = file.readline().strip()
        parts = line.split(',')
        if parts[0] == "leaf":
            tree.prediction = float(parts[1])
        else:
            tree.split_feature = int(parts[1])
            tree.split_value = float(parts[2])
            tree.left = ImprovedTree()
            tree.right = ImprovedTree()
            self._load_tree(tree.left, file)
            self._load_tree(tree.right, file)



if __name__ == "__main__":
    # Ensure the script runs only when executed directly
    print("Starting the training script...")

    #/ Punto di inizio del tempo
    start_time = time.time()

    #/ apro il dataset
    df = pd.read_csv("/dataset/filtered_dataset_20241118.csv")

    labels = df.pop("EOSS")


    #/ K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)
    fold = 1


    for train_index, val_index in kf.split(df, labels):
        x_train, x_test = df.values[train_index], df.values[val_index]
        y_train, y_test = labels.values[train_index], labels.values[val_index]

        model = SimpleXGBoost(n_estimators=500, learning_rate=0.1)
        model.fit(x_train, y_train)
        model.save_model(f"model_fold_{fold}.txt")
        pred2 = model.predict(x_test)
