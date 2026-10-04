"""
Machine Learning Model Architectures
Project: Smart Bike Rental Demand Analysis and Prediction
File: models.py
"""

import numpy as np

class LinearRegressionModel:
    """
    Linear Regression using Ordinary Least Squares (OLS) closed-form solution:
    theta = (X^T * X)^(-1) * X^T * y
    """
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Add intercept column of ones
        X_b = np.c_[np.ones(X.shape[0]), X]
        # Solve least squares
        theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.intercept = float(theta[0])
        self.weights = theta[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = self.intercept + X @ self.weights
        # Demand cannot be negative
        return np.maximum(0, preds)


class DecisionNode:
    """Node structure for Regression Decision Tree."""
    __slots__ = ('feature', 'threshold', 'left', 'right', 'value')
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressionModel:
    """Single Regression Tree with variance reduction splitting."""
    def __init__(self, max_depth=10, min_samples_split=15, max_features=8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.all(y == y[0]):
            return DecisionNode(value=float(np.mean(y)))

        # Subsample features for diversity (Random Forest principle)
        feat_indices = np.random.choice(n_features, size=min(self.max_features, n_features), replace=False)
        best_feat, best_thresh, best_gain = None, None, -1.0
        current_variance = np.var(y) * n_samples

        for feat in feat_indices:
            col_vals = X[:, feat]
            thresholds = np.unique(np.percentile(col_vals, np.linspace(10, 90, 9)))
            for t in thresholds:
                left_mask = col_vals <= t
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left
                if n_left < 5 or n_right < 5:
                    continue
                var_l = np.var(y[left_mask]) * n_left
                var_r = np.var(y[~left_mask]) * n_right
                gain = current_variance - (var_l + var_r)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = t

        if best_feat is None:
            return DecisionNode(value=float(np.mean(y)))

        left_mask = X[:, best_feat] <= best_thresh
        left_branch = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_branch = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return DecisionNode(feature=best_feat, threshold=best_thresh, left=left_branch, right=right_branch)

    def _predict_sample(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        return self._predict_sample(node.right, x)

    def predict(self, X):
        return np.array([self._predict_sample(self.root, row) for row in X])


class RandomForestRegressionModel:
    """
    Random Forest Regressor combining multiple decision trees
    using bootstrap aggregation (bagging) and random feature subsets.
    """
    def __init__(self, n_estimators=15, max_depth=10, min_samples_split=15, max_features=8, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_b, y_b = X[bootstrap_indices], y[bootstrap_indices]
            tree = DecisionTreeRegressionModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_b, y_b)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        all_tree_preds = np.array([tree.predict(X) for tree in self.trees])
        mean_preds = np.mean(all_tree_preds, axis=0)
        return np.maximum(0, mean_preds)
