# Use SVM to construct decision boundary for our neural network (cf. Figure 6.2)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# [1] Data Pre-Processing

x1 = [0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7]
x2 = [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]
X = np.column_stack((x1, x2))

arr1 = np.ones(5)
arr2 = np.zeros(5)
y_top = np.concatenate((arr1, arr2), axis=0)
y_bottom = np.concatenate((arr2, arr1), axis=0)
y = np.vstack((y_top, y_bottom)).T  

# Flatten y (labeled target variable) for SVM to predict classification
y = np.array([1,1,1,1,1,0,0,0,0,0])

# [2] Train an SVM model

# Create and train the SVM model (using a linear kernel)
classifier = svm.SVC(kernel='poly', C=1.0)
classifier.fit(X, y)

# [3] Plotting the decision boundary

# Create meshgrid
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict class labels 
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)

# Plot the training data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.bwr)
plt.title("SVM Decision Boundary (Cf. Figure 6.2, Higham (2019))")
plt.grid(False)
plt.show()



