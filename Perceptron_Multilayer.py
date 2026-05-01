import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11

class MLP:
    """
    Multilayer Perceptron with arbitrary architecture.
    Activation: tanh for all layers.
    Training: online backpropagation (one sample per step).
    Architecture example: [2, 4, 1] = 2 inputs, 4 hidden, 1 output.
    """

    def __init__(self, architecture, learning_rate=0.1, max_epochs=5000, beta=1.0):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.beta = beta
        self.arch = architecture
        self.n_layers = len(architecture) - 1
        # W[l]: shape (arch[l+1], arch[l]+1) — +1 for bias
        self.W = [
            np.random.uniform(-1, 1, size=(architecture[l+1], architecture[l] + 1))
            for l in range(self.n_layers)
        ]
        self.loss_history = []

    def _g(self, h):         return np.tanh(self.beta * h)
    def _gp(self, g):        return self.beta * (1 - g**2)  # g' expressed via g
    def _bias(self, a):      return np.concatenate([[1.0], a])

    def _forward(self, x):
        """Forward pass. Returns list of (h, g) per layer."""
        layers = []  # (h, a) per layer
        a = x
        for l in range(self.n_layers):
            h = self.W[l] @ self._bias(a)
            a = self._g(h)
            layers.append((h, a))
        return layers

    def predict_one(self, x):
        return self._forward(x)[-1][1]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def _backward(self, x, zeta):
        layers = self._forward(x)
        zeta = np.atleast_1d(zeta)
        n = self.n_layers
        deltas = [None] * n

        # Output layer delta
        h_out, a_out = layers[-1]
        deltas[-1] = (zeta - a_out) * self._gp(a_out)

        # Hidden layers — backpropagate
        for l in range(n - 2, -1, -1):
            h_l, a_l = layers[l]
            # W[l+1] without bias column
            deltas[l] = self._gp(a_l) * (self.W[l+1][:, 1:].T @ deltas[l+1])

        # Compute gradients
        grads = []
        for l in range(n):
            a_in = x if l == 0 else layers[l-1][1]
            grads.append(np.outer(deltas[l], self._bias(a_in)))

        return grads

    def fit(self, X, y, verbose_every=2000):
        p = X.shape[0]
        y = y.ravel()
        for epoch in range(self.max_epochs):
            idx = np.random.randint(p)
            for l, grad in enumerate(self._backward(X[idx], y[idx])):
                self.W[l] += self.lr * grad
            preds = self.predict(X)
            sse = 0.5 * np.sum((y - preds.ravel())**2)
            self.loss_history.append(sse)
            if verbose_every and (epoch+1) % verbose_every == 0:
                print(f"  Epoch {epoch+1:5d} | SSE = {sse:.4f}")
        return self