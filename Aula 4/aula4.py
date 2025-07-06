import numpy as np
import matplotlib.pyplot as plt
from generate import generate_synthetic_data

class KernelSVM:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, lr=0.01, max_iter=1000, tol=1e-4):
        self.C = C  # Termo de regularizacao
        self.kernel = kernel  # Funcao kernel ('rbf', 'poly', 'linear')
        self.gamma = gamma  # Parametro da funcao kernel (para RBF/poly)
        self.lr = lr  # Learning rate
        self.max_iter = max_iter  # Iteracoes
        self.tol = tol  # Tolerancia para parada
        self.alpha = None  # Multiplicadores de Lagrange
        self.b = 0  # Bias (termo linear)
        self.X_sv = None  # Support vectors
        self.y_sv = None  # Support vector labels
        self.alpha_sv = None  # Support vector alphas

    def _kernel(self, x1, x2):
        """Calcula as funcoes kernel"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** self.gamma
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Kernel desconhecido")

    def _compute_kernel_matrix(self, X):
        """Calcula a matriz de kernels: K_{ij} = K(x_i, x_j)"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel(X[i], X[j])
        return K

    def _objective(self, alpha, K, y):
        """Funcao objetivo (Lagrangiana)"""
        return np.sum(alpha) - 0.5 * np.sum(alpha * alpha * y * y * K)

    def _clip_alpha(self, alpha):
        """Projeta alpha para [0, C]"""
        return np.clip(alpha, 0, self.C)

    def _project_alpha(self, alpha, y):
        """Enforca restricao: sum(alpha_i * y_i) = 0"""
        alpha -= np.sum(alpha * y) / np.sum(y * y) * y
        return self._clip_alpha(alpha)

    def fit(self, X, y):
        """Treina SVM com gradiente ascendente"""
        n_samples = X.shape[0]
        self.alpha = np.random.random(n_samples)
        y = y.astype(float)  # Faz com que os valores de y sejam float

        # Calcula matriz de kernels
        K = self._compute_kernel_matrix(X)

        # Gradient ascent
        prev_obj = -np.inf
        for _ in range(self.max_iter):
            # Calcula gradiente
            grad = 1 - y * np.dot(K, self.alpha * y)

            # Update dos alphas
            self.alpha += self.lr * grad
            self.alpha = self._project_alpha(self.alpha, y)

            # Calcula Lagrangiana atual
            current_obj = self._objective(self.alpha, K, y)

            # Checa se ja convergiu
            if np.abs(current_obj - prev_obj) < self.tol:
                print("Brekou")
                break
            prev_obj = current_obj

        # Pega os support vectors (alpha_i > 0)
        sv_indices = self.alpha > 1e-2
        self.X_sv = X[sv_indices]
        self.y_sv = y[sv_indices]
        self.alpha_sv = self.alpha[sv_indices]

        # Calcula o bias
        self.b = np.mean(
            self.y_sv - np.sum(
                self.alpha_sv * self.y_sv * self._compute_kernel_matrix(self.X_sv),
                axis=1
            )
        )

    def predict(self, X):
        """Predict using the trained SVM"""
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for a, sv_y, sv_x in zip(self.alpha_sv, self.y_sv, self.X_sv):
                s += a * sv_y * self._kernel(X[i], sv_x)
            y_pred[i] = np.sign(s + self.b)
        return y_pred

    def plot_decision_boundary(self, X, y):
        """Plot the data points and decision boundary"""
        plt.figure(figsize=(10, 6))

        # Plot training data
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', label='Data')

        # Create grid for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict on grid
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], cmap='bwr')
        plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles='dashed')

        # Highlight support vectors
        if self.X_sv is not None:
            plt.scatter(
                self.X_sv[:, 0], self.X_sv[:, 1],
                facecolors='none', edgecolors='k', s=100,
                label='Support Vectors'
            )

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Kernel SVM Decision Boundary')
        plt.legend()
        plt.show()



def train_test_split(X, y, test_size=0.3, random_state=None):
    """Simple train-test split without sklearn"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def accuracy_score(y_true, y_pred):
    """Calculate accuracy without sklearn"""
    return np.mean(y_true == y_pred)

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_synthetic_data(data_type='xor')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train kernel SVM
    svm = KernelSVM(C=1.0, kernel='rbf', gamma=0.6, lr=0.01, max_iter=10)
    svm.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Plot decision boundary
    svm.plot_decision_boundary(X_train, y_train)