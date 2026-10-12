# EXPERIMENT 4: ARTIFICIAL NEURAL NETWORK
# BACKPROPAGATION

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

# -----------------------------
# LOSS GRAPH
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    model.loss_curve_,
    marker='o'
)

plt.title("Backpropagation Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.grid(True)

plt.show()
