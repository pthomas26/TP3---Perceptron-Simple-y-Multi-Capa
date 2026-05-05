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

    def __init__(self, architecture, learning_rate=0.1, max_epochs=5000, beta=1.0,
                 optimizer='sgd'):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.beta = beta
        self.arch = architecture
        self.optimizer = optimizer
        self.n_layers = len(architecture) - 1
        # W[l]: shape (arch[l+1], arch[l]+1) — +1 for bias
        self.W = [
            np.random.uniform(-1, 1, size=(architecture[l+1], architecture[l] + 1))
            for l in range(self.n_layers)
        ]
        self.loss_history = []

        if optimizer in ('momentum', 'rmsprop'):
            self.vW = [np.zeros((architecture[l+1], architecture[l])) for l in range(self.n_layers)]
            self.vb = [np.zeros(architecture[l+1])                    for l in range(self.n_layers)]
        elif optimizer == 'adam':
            self.mW = [np.zeros((architecture[l+1], architecture[l])) for l in range(self.n_layers)]
            self.mb = [np.zeros(architecture[l+1])                    for l in range(self.n_layers)]
            self.vW = [np.zeros((architecture[l+1], architecture[l])) for l in range(self.n_layers)]
            self.vb = [np.zeros(architecture[l+1])                    for l in range(self.n_layers)]
            self.t  = 0

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
            idx   = np.random.randint(p)
            grads = self._backward(X[idx], y[idx])

            if self.optimizer == 'sgd':
                for l, grad in enumerate(grads):
                    self.W[l] += self.lr * grad

            elif self.optimizer == 'momentum':
                gamma = 0.9
                for l, grad in enumerate(grads):
                    self.vW[l] = gamma * self.vW[l] + self.lr * grad[:, 1:]
                    self.vb[l] = gamma * self.vb[l] + self.lr * grad[:, 0]
                    self.W[l][:, 1:] += self.vW[l]
                    self.W[l][:, 0]  += self.vb[l]

            elif self.optimizer == 'rmsprop':
                rho, eps = 0.9, 1e-8
                for l, grad in enumerate(grads):
                    gW, gb = grad[:, 1:], grad[:, 0]
                    self.vW[l] = rho * self.vW[l] + (1 - rho) * gW ** 2
                    self.vb[l] = rho * self.vb[l] + (1 - rho) * gb ** 2
                    self.W[l][:, 1:] += self.lr * gW / (np.sqrt(self.vW[l]) + eps)
                    self.W[l][:, 0]  += self.lr * gb / (np.sqrt(self.vb[l]) + eps)

            elif self.optimizer == 'adam':
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self.t += 1
                for l, grad in enumerate(grads):
                    gW, gb = grad[:, 1:], grad[:, 0]
                    self.mW[l] = beta1 * self.mW[l] + (1 - beta1) * gW
                    self.mb[l] = beta1 * self.mb[l] + (1 - beta1) * gb
                    self.vW[l] = beta2 * self.vW[l] + (1 - beta2) * gW ** 2
                    self.vb[l] = beta2 * self.vb[l] + (1 - beta2) * gb ** 2
                    mW_hat = self.mW[l] / (1 - beta1 ** self.t)
                    mb_hat = self.mb[l] / (1 - beta1 ** self.t)
                    vW_hat = self.vW[l] / (1 - beta2 ** self.t)
                    vb_hat = self.vb[l] / (1 - beta2 ** self.t)
                    self.W[l][:, 1:] += self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
                    self.W[l][:, 0]  += self.lr * mb_hat / (np.sqrt(vb_hat) + eps)
            preds = self.predict(X)
            sse = 0.5 * np.sum((y - preds.ravel())**2)
            self.loss_history.append(sse)
            if verbose_every and (epoch+1) % verbose_every == 0:
                print(f"  Epoch {epoch+1:5d} | SSE = {sse:.4f}")
        return self
    



if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n_features, n_classes = 4, 1

    X_all = rng.standard_normal((300, n_features))
    y_all = np.sign(X_all[:, 0] + X_all[:, 1]).astype(float)  # {-1, 1}

    X_tr, y_tr   = X_all[:240], y_all[:240]
    X_val, y_val = X_all[240:], y_all[240:]

    for opt in ['sgd', 'momentum', 'rmsprop', 'adam']:
        m = MLP([n_features, 64, n_classes], optimizer=opt, max_epochs=3)
        m.fit(X_tr, y_tr, verbose_every=0)
        acc = np.mean(np.sign(m.predict(X_val).ravel()) == y_val)
        print(f"{opt}: OK — val acc {acc:.3f}")