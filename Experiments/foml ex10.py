import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
# Load dataset
digits = load_digits()
X = digits.data
y = digits.target
# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Plot PCA results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title("PCA - Digit Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


