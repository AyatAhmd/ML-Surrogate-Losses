"""Linear logistic regression trained with logistic loss and L2 regularization.

Implemented from scratch: no scikit-learn estimators or shared classifier class.
"""
import numpy as np
import pandas as pd


class LogisticRegressionClassifier:
    """Fit a linear score w.T @ x + b by full-batch (sub)gradient descent.

    Objective: mean logistic loss + (l2 / 2) * ||w||^2.
    The intercept b is not regularized.
    """

    def __init__(self, epochs=240, learning_rate=0.6, l2=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        if self.epochs <= 0 or self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("Invalid training settings")
        self.history_ = None

    def decision_function(self, X):
        """Return real-valued linear scores, before converting to classes."""
        return np.asarray(X) @ self.coef_ + self.intercept_

    def predict(self, X):
        """Assign +1 when score >= 0, otherwise -1."""
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def error(self, X, y):
        """Fraction of incorrect class predictions (0-1 error)."""
        return float(np.mean(self.predict(X) != y))

    def surrogate_loss(self, X, y):
        """Mean logistic loss, without the L2 penalty."""
        margin = np.asarray(y) * self.decision_function(X)
        # Logistic loss written with logaddexp for numerical stability.
        losses = np.logaddexp(0.0, -margin)
        return float(np.mean(losses))

    def objective(self, X, y):
        """Training objective: mean loss plus L2 weight penalty."""
        return self.surrogate_loss(X, y) + 0.5 * self.l2 * float(self.coef_ @ self.coef_)

    def _gradient(self, X, y):
        """Gradient of the training objective with respect to weights and bias."""
        margin = y * self.decision_function(X)
        # Derivative d(loss)/d(margin) = -1/(1 + exp(margin)).
        dloss_dmargin = -np.exp(-np.logaddexp(0.0, margin))
        # Chain rule: margin = y * (X @ w + b).
        g = dloss_dmargin * y
        grad_w = (X.T @ g) / len(y) + self.l2 * self.coef_
        grad_b = float(g.mean())  # No regularization on the intercept.
        return grad_w, grad_b

    def _record(self, epoch, X_train, y_train, X_test, y_test):
        """Record metrics before training (epoch 0) and after each epoch."""
        row = {"epoch": epoch,
               "train_surrogate": self.surrogate_loss(X_train, y_train),
               "train_error": self.error(X_train, y_train),
               "objective": self.objective(X_train, y_train),
               "weight_norm": float(np.linalg.norm(self.coef_))}
        if X_test is not None:
            row.update(test_surrogate=self.surrogate_loss(X_test, y_test),
                       test_error=self.error(X_test, y_test))
        return row

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """Train using the complete training set in each epoch.

        The test set, when provided, is used ONLY for recording test metrics.
        Backtracking halves the step until the training objective does not rise.
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        if X_train.ndim != 2 or len(X_train) != len(y_train) or not len(X_train):
            raise ValueError("Invalid training dimensions")
        if not np.isfinite(X_train).all() or not set(np.unique(y_train)).issubset({-1, 1}):
            raise ValueError("Invalid training data")
        if (X_test is None) != (y_test is None):
            raise ValueError("Provide X_test and y_test together")

        self.coef_ = np.zeros(X_train.shape[1], dtype=float)
        self.intercept_ = 0.0
        self.history_ = [self._record(0, X_train, y_train, X_test, y_test)]

        for epoch in range(1, self.epochs + 1):
            # Full-batch (sub)gradient: uses every training observation.
            grad_w, grad_b = self._gradient(X_train, y_train)
            old_w, old_b = self.coef_.copy(), self.intercept_
            old_obj = self.history_[-1]["objective"]
            step = self.learning_rate

            # Backtracking
            for _ in range(35):
                self.coef_ = old_w - step * grad_w
                self.intercept_ = old_b - step * grad_b
                if self.objective(X_train, y_train) <= old_obj + 1e-12:
                    break
                step *= 0.5
            else:
                self.coef_, self.intercept_ = old_w, old_b

            self.history_.append(self._record(epoch, X_train, y_train,
                                             X_test, y_test))

        self.history_ = pd.DataFrame(self.history_)
        return self
