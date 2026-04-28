import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11

class StepPerceptron:
    """Simple step perceptron. Bias treated as extra weight with input=1."""

    def __init__(self, n_features, learning_rate=0.1, max_epochs=1000):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = np.random.uniform(-1, 1, size=n_features + 1)
        self.history = []

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _activation(self, h):
        return np.where(h >= 0, 1, -1)

    def predict(self, X):
        return self._activation(self._add_bias(X) @ self.w)

    def fit(self, X, y):
        X_b = self._add_bias(X)
        p = X_b.shape[0]
        best_w, best_error = self.w.copy(), p

        for epoch in range(self.max_epochs):
            idx = np.random.randint(p)
            o = self._activation(X_b[idx] @ self.w)
            self.w += self.lr * (y[idx] - o) * X_b[idx]

            error = np.sum(self._activation(X_b @ self.w) != y)
            self.history.append(error)
            if error < best_error:
                best_error, best_w = error, self.w.copy()
            if error == 0:
                print(f"  Converged at epoch {epoch+1}")
                break

        self.w = best_w
        return self
    