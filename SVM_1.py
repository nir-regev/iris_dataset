# Re-importing necessary libraries for SVM evaluation after reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Training models with different kernels
models = {
    'Linear': SVC(kernel='linear'),
    'RBF': SVC(kernel='rbf', gamma='scale'),
    'Polynomial': SVC(kernel='poly', degree=3, gamma='scale')
}

# Training and predictions
predictions = {}
accuracies = {}
for kernel, model in models.items():
    model.fit(X_train, y_train)
    predictions[kernel] = model.predict(X_test)
    accuracies[kernel] = accuracy_score(y_test, predictions[kernel])

# Plotting confusion matrices
plt.figure(figsize=(18, 6))
for i, (kernel, prediction) in enumerate(predictions.items(), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt="d", cmap="Blues", annot_kws={"color": "black"})
    plt.title(f'{kernel} Kernel\nAccuracy: {accuracies[kernel]:.2f}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')


plt.savefig('heatmap1.png')
plt.tight_layout()
plt.show()

# Displaying accuracy scores comparison
plt.figure(figsize=(8, 4))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Comparison of SVM Kernels on Iris Dataset')
plt.ylim(0.9, 1.0)
plt.show()

