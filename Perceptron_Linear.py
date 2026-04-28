import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11

class LinearPerceptron:
    """ADALINE: identity activation, SSE loss, delta rule update."""

    def __init__(self, n_features, learning_rate=0.01, max_epochs=2000):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = np.random.uniform(-1, 1, size=n_features + 1)
        self.loss_history = []

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def predict(self, X):
        return self._add_bias(X) @ self.w

    def fit(self, X, y, verbose_every=500):
        X_b = self._add_bias(X)
        p = X_b.shape[0]
        for epoch in range(self.max_epochs):
            idx = np.random.randint(p)
            o = X_b[idx] @ self.w
            self.w += self.lr * (y[idx] - o) * X_b[idx]
            sse = 0.5 * np.sum((y - X_b @ self.w) ** 2)
            self.loss_history.append(sse)
            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  Epoch {epoch+1:4d} | SSE = {sse:.4f}")
        return self