import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, n_features=10, centers=2, random_state=41)
print(X.shape)
feature1, feature2, feature3 = int(np.random.uniform(0,10)), int(np.random.uniform(0,10)),\
                               int(np.random.uniform(0,10))
X = X[:,[feature1,feature2,feature3]]

class F:
    @staticmethod
    def sign(x):
        return np.where(x==0,-1,1)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=1000) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_ = []  # storing the number of misclassifications in each epoch

    def fit(self,X,y):
        n_samples, n_features = X.shape
        # starting weights and bias equal zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            errors = 0
            for idx in range(n_samples):
                linear_output = np.dot(X[idx], self.weights) + self.bias  # w^T x + b
                y_pred = self._unit_step(linear_output)
                if y[idx] != y_pred: # misclassfied
                    update = self.learning_rate * y[idx]
                    self.weights += update * X[idx]
                    self.bias += update
                    errors += 1
            self.errors_.append(errors)
            # if no errors, convergence achieved
            if errors == 0:
                print(f"Converged after {epoch+1} epochs")
                break

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X : array-like, shape = [n_samples, n_features]

        Returns:
        array, shape = [n_samples]
            Predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias  # w^T * x + b
        return self._unit_step(linear_output)

    def _unit_step(self, x):
        return np.where(x >= 0, 1, -1)

y = F.sign(y)
perceptron = Perceptron(0.01,1000)   
perceptron.fit(X,y)            

from matplotlib.colors import ListedColormap

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500),
                         np.linspace(z_min, z_max, 500))

grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

Z = perceptron.predict(grid)
Z = Z.reshape(xx.shape)

def generate_ab_class(n_points=100):
    """Generate Data Class_A & Class_B"""
    class_A = []
    class_B = []
    for i in range(len(X)):
        if (y[i] == 1):
            class_A.append(X[i])
        elif (y[i] == -1):
            class_B.append(X[i])
    return np.array(class_A), np.array(class_B)

class_A, class_B = generate_ab_class()

# Visualize Perceptron
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
fig = plt.figure(figsize=(10, 6))
ax = plt.axes (projection='3d')
ax.grid()
ax.scatter(class_A[:, 0], class_A[:, 1],  class_A[:, 2], color='red', marker='o', label='Class A')
ax.scatter(class_B[:, 0], class_B[:, 1],  class_B[:, 2], color='blue', marker='s', label='Class B')

ax.set_xlabel('x['+str(feature1)+']')
ax.set_ylabel('x['+str(feature2)+']')
ax.set_zlabel('x['+str(feature3)+']')

#plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
if perceptron.weights[2] != 0:
    x_vals = np.array([x_min, x_max])
    y_vals = np.array([y_min, y_max])
    z_vals = -(perceptron.weights[0] * x_vals + perceptron.weights[1] * y_vals + perceptron.bias) / perceptron.weights[2]
    ax.plot3D(x_vals, y_vals, z_vals, 'k--', label='Decision Boundary')
elif perceptron.weights[1] != 0:
    x_vals = np.array([x_min, x_max])
    y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
    plt.plot3D(x_vals, y_vals, 'k--', label='Decision Boundary')
else:
    x_val = -perceptron.bias / perceptron.weights[0]
    plt.axvline(x=x_val, color='k', linestyle='--', label='Decision Boundary')

"""
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
if perceptron.weights[1] != 0:
    x_vals = np.array([x_min, x_max])
    y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
else:
    x_val = -perceptron.bias / perceptron.weights[0]
    plt.axvline(x=x_val, color='k', linestyle='--', label='Decision Boundary')


plt.scatter(class_A[:, 0], class_A[:, 1],  class_A[:, 2], color='red', marker='o', label='Class A')
plt.scatter(class_B[:, 0], class_B[:, 1],  class_B[:, 2], color='blue', marker='s', label='Class B')
plt.xlabel('X['+str(feature1)+']')
plt.ylabel('X['+str(feature2)+']')
"""
plt.title('Perceptron Decision Boundary and Decision Regions')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(perceptron.errors_) + 1,1), perceptron.errors_[::1], marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.title('Perceptron Learning Progress')
plt.grid(True)
plt.show()
print (len(perceptron.errors_))