"""Small, notebook-first experiment interface (no command line or output folders)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import Dataset, StratifiedSplitter, TrainOnlyStandardizer
from .svm import LinearSVM
from .logistic_regression import LogisticRegressionClassifier
from .least_squares import LeastSquaresClassifier

# One separate class for each loss. Dictionary keys preserve the old experiment labels.
CLASSIFIERS = {
    "hinge": LinearSVM,
    "logistic": LogisticRegressionClassifier,
    "squared": LeastSquaresClassifier,
}


@dataclass
class ExperimentResult:
    dataset: Dataset
    split: object
    scaler: TrainOnlyStandardizer
    X_train: np.ndarray
    X_test: np.ndarray
    models: dict
    summary: pd.DataFrame

    @property
    def histories(self):
        return {name: model.history_ for name, model in self.models.items()}


class ExperimentRunner:
    """Reproducible train/test comparison of all three convex surrogate losses."""

    def __init__(self, seed=42, test_size=0.25, epochs=180, learning_rate=0.6):
        self.seed = seed
        self.splitter = StratifiedSplitter(test_size=test_size, random_state=seed)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def prepare(self, dataset):
        split = self.splitter.split(dataset)
        scaler = TrainOnlyStandardizer().fit(split.X_train)
        return split, scaler, scaler.transform(split.X_train), scaler.transform(split.X_test)

    def _fit(self, loss, X_train, y_train, X_test, y_test,
             l2=0.01, epochs=None):
        model = CLASSIFIERS[loss](
            epochs=self.epochs if epochs is None else epochs,
            learning_rate=self.learning_rate,
            l2=l2,
        )
        return model.fit(X_train, y_train, X_test, y_test)

    def run(self, dataset, l2=0.01):
        split, scaler, X_train, X_test = self.prepare(dataset)
        models = {}
        rows = []
        for name in CLASSIFIERS:
            model = self._fit(name, X_train, split.y_train, X_test, split.y_test, l2=l2)
            models[name] = model
            rows.append({'dataset': dataset.name, 'loss': name,
                         'train_error': model.error(X_train, split.y_train),
                         'test_error': model.error(X_test, split.y_test),
                         'train_surrogate': model.surrogate_loss(X_train, split.y_train),
                         'test_surrogate': model.surrogate_loss(X_test, split.y_test),
                         'generalization_gap': model.error(X_test, split.y_test) - model.error(X_train, split.y_train)})
        return ExperimentResult(dataset, split, scaler, X_train, X_test, models, pd.DataFrame(rows))

    def regularization(self, dataset, penalties=(0., 0.001, 0.01, 0.1, 1.0), epochs=120):
        split, scaler, X_train, X_test = self.prepare(dataset)
        rows = []
        for strength in penalties:
            for name in CLASSIFIERS:
                model = self._fit(name, X_train, split.y_train, X_test, split.y_test,
                                  l2=strength, epochs=epochs)
                rows.append({'dataset': dataset.name, 'loss': name, 'l2': strength,
                             'train_error': model.error(X_train, split.y_train),
                             'test_error': model.error(X_test, split.y_test),
                             'weight_norm': np.linalg.norm(model.coef_)})
        return pd.DataFrame(rows)
