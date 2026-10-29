"""Reproducible datasets, leakage-safe splits, and train-only standardization."""
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen, Request
import io
import zipfile

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-",
    "C%", "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L",
    "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)",
    "nCIR", "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d",
    "B04[C-Br]", "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]",
    "nHDon", "SpMax_B(m)", "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX"
]


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        if self.X.ndim != 2 or self.X.shape[0] != len(self.y):
            raise ValueError("X/y shape mismatch")
        if not np.isfinite(self.X).all() or set(np.unique(self.y)) != {-1, 1}:
            raise ValueError("Finite features and both {-1,1} labels required")


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


class SyntheticDataGenerator:
    """Two-dimensional controlled data-generating processes."""

    def __init__(self, n_samples=600, random_state=42):
        self.n_samples = int(n_samples)
        self.random_state = int(random_state)
        if self.n_samples < 30:
            raise ValueError("Need at least 30 observations")

    def linearly_separable(self):
        rng = np.random.default_rng(self.random_state)
        chunks = []
        while sum(len(c) for c in chunks) < self.n_samples:
            x = rng.uniform(-3, 3, size=(self.n_samples * 2, 2))
            signed = x[:, 0] + 0.7 * x[:, 1] - 0.1
            chunks.append(x[np.abs(signed) > 0.65])
        X = np.concatenate(chunks)[:self.n_samples]
        y = np.where(X[:, 0] + 0.7 * X[:, 1] - 0.1 >= 0, 1, -1)
        return Dataset("synthetic_separable", X, y, ["x1", "x2"])

    def nonseparable(self, flip_fraction=0.17):
        rng = np.random.default_rng(self.random_state + 1)
        X = rng.uniform(-3, 3, size=(self.n_samples, 2))
        boundary = X[:, 0] + 0.85 * np.sin(2.2 * X[:, 1]) - 0.15
        y = np.where(boundary >= 0, 1, -1)
        flip = rng.choice(self.n_samples, size=int(round(flip_fraction * self.n_samples)), replace=False)
        y[flip] *= -1
        return Dataset("synthetic_nonseparable", X, y, ["x1", "x2"])

    @staticmethod
    def save(dataset: Dataset, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(dataset.X, columns=dataset.feature_names)
        df["target"] = dataset.y
        df.to_csv(path, index=False, float_format="%.10g")


class QSARDataLoader:
    """Load original UCI biodeg.csv; fetch official ZIP when it is absent."""
    ZIP_URL = "https://archive.ics.uci.edu/static/public/254/qsar%2Bbiodegradation.zip"
    CSV_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "biodeg.csv"

    def download(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        errors = []
        for url in [self.ZIP_URL, self.CSV_URL]:
            try:
                with urlopen(Request(url, headers={"User-Agent": "Academic-Research/1.0"}), timeout=30) as response:
                    payload = response.read()
                if payload[:2] == b"PK":
                    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                        candidates = [n for n in archive.namelist() if n.lower().endswith("biodeg.csv")]
                        if not candidates:
                            raise ValueError("UCI ZIP does not contain biodeg.csv")
                        payload = archive.read(candidates[0])
                if not payload or b";" not in payload:
                    raise ValueError("Downloaded payload is not semicolon-delimited CSV")
                self.path.write_bytes(payload)
                return self.path
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        raise FileNotFoundError(
            "Unable to download QSAR. Place the ORIGINAL UCI biodeg.csv in Data/. "
            "Attempts: " + " | ".join(errors)
        )

    def load(self, download=True):
        if not self.path.exists():
            if not download:
                raise FileNotFoundError(f"Missing original UCI file: {self.path}")
            self.download()
        df = pd.read_csv(self.path, sep=";", header=None)
        if df.shape[1] != 42:
            raise ValueError(f"Expected 42 UCI columns, found {df.shape[1]}. Check the raw file.")
        df.columns = FEATURE_NAMES + ["target"]
        if not set(df["target"].unique()).issubset({"RB", "NRB"}):
            raise ValueError("Unexpected target names (expected RB/NRB)")
        df[FEATURE_NAMES] = df[FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
        df = df.drop_duplicates().reset_index(drop=True)
        # Identical features with different targets require inspection rather than silent collapse.
        conflicts = df.groupby(FEATURE_NAMES, dropna=False)["target"].nunique()
        if (conflicts > 1).any():
            raise ValueError("Identical descriptor vectors with conflicting labels found")
        return Dataset("qsar_biodegradation", df[FEATURE_NAMES].to_numpy(),
                       df["target"].map({"RB": 1, "NRB": -1}).to_numpy(), FEATURE_NAMES)


class StratifiedSplitter:
    def __init__(self, test_size=0.25, random_state=42):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be in (0,1)")
        self.test_size = test_size
        self.random_state = random_state

    def split(self, dataset: Dataset) -> Split:
        rng = np.random.default_rng(self.random_state)
        tr, te = [], []
        for label in (-1, 1):
            idx = rng.permutation(np.flatnonzero(dataset.y == label))
            n_test = max(1, min(len(idx) - 1, round(len(idx) * self.test_size)))
            te.extend(idx[:n_test])
            tr.extend(idx[n_test:])
        tr, te = rng.permutation(tr), rng.permutation(te)
        return Split(dataset.X[tr], dataset.X[te], dataset.y[tr], dataset.y[te], tr, te)


class TrainOnlyStandardizer:
    """Means and standard deviations estimated ONLY from the training partition."""
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.
        return self

    def transform(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("Must call fit() first")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
