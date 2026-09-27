import numpy as np
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_svmlight_file


LIBSVM_BINARY_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"

# SHA-256 of the official scaled files, with CRLF normalized to LF so that
# the existing Windows copies can also serve as a valid local cache.
DATASET_SHA256 = {
    "sonar_scale": "037cfd45bb601292b0ecfa6d21ac0ebe61564d85cfe6d4dd61f221b9157174c5",
    "heart_scale": "5defa0a4c4c5bdaf3f55ae3828310252e8565c13ee37ce279e0b86d82e7f4ce9",
    "australian_scale": "86799bcc4dbcd41929a514fa181d1c1a40e01f009dee139d4141a194d7ecd9e3",
    "diabetes_scale": "0c07eb4c49e7a8ffb9c9f25095ac3022df2ca85b0dcb7d294c3ddea69f392cba",
    "fourclass_scale": "e27b003cd53d3141f8266b7e1db0d43e69f2824cc94c1d885fb27c8f2145a511",
    "german.numer_scale": "97637cc6b994bd3fa78999c771bc044bea7a0cdf91d43b6ecfd10f7e7c4ebc82",
}


def _dataset_hash(contents):
    return hashlib.sha256(contents.replace(b"\r\n", b"\n")).hexdigest()


def fetch_dataset(name, cache_dir=None):
    """Fetch an official LIBSVM binary dataset, reusing a verified cache."""
    if name not in DATASET_SHA256:
        raise ValueError(f"Unknown LIBSVM dataset: {name}")

    cache = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent / "data"
    path = cache / name
    expected_hash = DATASET_SHA256[name]
    if path.is_file() and _dataset_hash(path.read_bytes()) == expected_hash:
        return path

    url = LIBSVM_BINARY_URL + name
    try:
        with urlopen(url, timeout=30) as response:
            contents = response.read()
    except Exception as exc:
        raise RuntimeError(f"Could not download {name} from {url}") from exc
    if _dataset_hash(contents) != expected_hash:
        raise ValueError(f"Checksum mismatch for LIBSVM dataset {name} from {url}")

    cache.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=cache, prefix=f".{name}.",
                                         suffix=".tmp", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(contents)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return path


def load_dataset(name, cache_dir=None):
    path = fetch_dataset(name, cache_dir=cache_dir)
    X, y = load_svmlight_file(path)

    X = X.toarray()

    labels = np.unique(y)

    if len(labels) != 2:
        raise ValueError(
            f"This is supposed to be binary classification, not for the: {labels} classes"
        )

    y = np.where(
        y == labels[0],
        -1.0,
        1.0
    )

    return X, y


def gamma_med(X):
    D2 = pairwise_distances(
        X,
        metric="sqeuclidean"
    )

    mask = ~np.eye(len(X), dtype=bool)

    distances = D2[mask]
    distances = distances[distances > 0]

    median_d2 = np.median(distances)
    gamma_med = 1.0 / median_d2

    return gamma_med


def build_dual_matrix(X, y, kernel='linear', gamma=None):
    if kernel == 'linear':
        K = X @ X.T

    elif kernel == 'rbf':
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        K = rbf_kernel(X, gamma=gamma)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    Q = (y[:, None] * y[None, :]) * K

    return Q, K


def diag_scaling(Q: np.ndarray, C: float):
    d = np.diag(Q)
    D_inv_sqrt = np.diag(1 / np.sqrt(d))

    H = D_inv_sqrt @ Q @ D_inv_sqrt
    a = D_inv_sqrt @ np.ones(Q.shape[0])

    r = C * np.sqrt(np.diag(Q))

    T = np.tril(H, -1)
    tau = svds(T, k=1, which="LM", return_singular_vectors=False)[0]

    return H, a, T, tau, r
