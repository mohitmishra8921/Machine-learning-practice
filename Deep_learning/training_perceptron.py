from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = Perceptron(
    max_iter=1000,    # Maximum number of epochs
    eta0=0.1,         # Learning rate
    random_state=42,  # For reproducibility
    tol=1e-3,         # Stop early if improvement is smaller than this
    shuffle=True      # Shuffle data each epoch
)


clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)*100
print(f"Accuracy: {accuracy:.2f}")