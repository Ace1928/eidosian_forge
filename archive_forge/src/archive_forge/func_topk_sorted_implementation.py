import numpy as np
from onnx.reference.op_run import OpRun
def topk_sorted_implementation(X, k, axis, largest):
    """See function `_kneighbors_reduce_func
    <https://github.com/scikit-learn/scikit-learn/blob/main/
    sklearn/neighbors/_base.py#L304>`_.
    """
    if isinstance(k, np.ndarray):
        if k.size != 1:
            raise RuntimeError(f'k must be an integer not {k!r}.')
        k = k[0]
    k = int(k)
    if len(X.shape) == 2 and axis == 1:
        sample_range = np.arange(X.shape[0])[:, None]
        if largest == 0:
            sorted_indices = np.argpartition(X, axis=axis, kth=k - 1)
            sorted_indices = sorted_indices[:, :k]
            sorted_indices = sorted_indices[sample_range, np.argsort(X[sample_range, sorted_indices])]
        else:
            sorted_indices = np.argpartition(-X, axis=axis, kth=k - 1)
            sorted_indices = sorted_indices[:, :k]
            sorted_indices = sorted_indices[sample_range, np.argsort(-X[sample_range, sorted_indices])]
        sorted_distances = X[sample_range, sorted_indices]
        return (sorted_distances, sorted_indices)
    sorted_indices = np.argsort(X, axis=axis)
    sorted_values = np.sort(X, axis=axis)
    if largest:
        sorted_indices = np.flip(sorted_indices, axis=axis)
        sorted_values = np.flip(sorted_values, axis=axis)
    ark = np.arange(k)
    topk_sorted_indices = np.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = np.take(sorted_values, ark, axis=axis)
    return (topk_sorted_values, topk_sorted_indices)