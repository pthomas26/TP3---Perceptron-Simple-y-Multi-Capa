import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11

class NonLinearPerceptron:
    """Non-linear simple perceptron. Supports tanh, sigmoid, relu activations."""

    ACTIVATIONS = {
        'tanh':    (lambda h, b: np.tanh(b * h),             lambda g, b: b * (1 - g**2)),
        'sigmoid': (lambda h, b: 1/(1+np.exp(-2*b*h)),       lambda g, b: 2*b*g*(1-g)),
        'relu':    (lambda h, b: np.maximum(0, h),            lambda g, b: (g > 0).astype(float)),
    }

    def __init__(self, n_features, learning_rate=0.01, max_epochs=3000,
                 activation='tanh', beta=1.0):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.beta = beta
        self.activation_name = activation
        self.g, self.g_prime = self.ACTIVATIONS[activation]
        self.w = np.random.uniform(-1, 1, size=n_features + 1)
        self.loss_history = []

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def predict(self, X):
        h = self._add_bias(X) @ self.w
        return self.g(h, self.beta)

    def fit(self, X, y, verbose_every=1000):
        X_b = self._add_bias(X)
        p = X_b.shape[0]
        for epoch in range(self.max_epochs):
            idx = np.random.randint(p)
            h = X_b[idx] @ self.w
            o = self.g(h, self.beta)
            grad = (y[idx] - o) * self.g_prime(o, self.beta)
            self.w += self.lr * grad * X_b[idx]
            preds = self.predict(X)
            self.loss_history.append(0.5 * np.sum((y - preds) ** 2))
            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  Epoch {epoch+1:5d} | SSE = {self.loss_history[-1]:.4f}")
        return self