import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.weights = None

    def sigmoid(self, z):
        """Apply the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """Compute the cost function."""
        epsilon = 1e-10  # Small constant to avoid log(0)
        predictions = self.sigmoid(np.dot(X, self.weights))
        cost = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
        return cost

    def fit(self, X, y):
        """Train the Linear Regression model using SGD."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # Convert y to a NumPy array for shuffling compatibility
        y = np.array(y)

        prev_cost = float('inf')

        for _ in range(self.max_iter):
            # Shuffle data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Update weights for each sample
            for i in range(n_samples):
                gradient = (self.sigmoid(np.dot(X[i], self.weights)) - y[i]) * X[i]
                self.weights -= self.learning_rate * gradient

            # Compute cost
            current_cost = self.compute_cost(X, y)
            if abs(prev_cost - current_cost) < self.tolerance:
                break
            prev_cost = current_cost


    

    def predict(self, X):
        """Make predictions."""
        probabilities = self.sigmoid(np.dot(X, self.weights))
        return (probabilities >= 0.5).astype(int)  # Convert probabilities to binary labels
