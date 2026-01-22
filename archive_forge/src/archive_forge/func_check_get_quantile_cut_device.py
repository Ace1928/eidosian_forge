import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def check_get_quantile_cut_device(tree_method: str, use_cupy: bool) -> None:
    """Check with optional cupy."""
    from pandas.api.types import is_categorical_dtype
    n_samples = 1024
    n_features = 14
    max_bin = 16
    dtypes = [np.float32] * n_features
    X, y, w = tm.make_regression(n_samples, n_features, use_cupy=use_cupy)
    Xyw: xgb.DMatrix = xgb.QuantileDMatrix(X, y, weight=w, max_bin=max_bin)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)
    Xyw = xgb.DMatrix(X, y, weight=w)
    xgb.train({'tree_method': tree_method, 'max_bin': max_bin}, Xyw)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)
    n_batches = 3
    n_samples_per_batch = 256
    it = tm.IteratorForTest(*tm.make_batches(n_samples_per_batch, n_features, n_batches, use_cupy), cache='cache')
    Xy: xgb.DMatrix = xgb.DMatrix(it)
    xgb.train({'tree_method': tree_method, 'max_bin': max_bin}, Xyw)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)
    n_categories = 32
    X, y = tm.make_categorical(n_samples, n_features, n_categories, False, sparsity=0.8)
    if use_cupy:
        import cudf
        import cupy as cp
        X = cudf.from_pandas(X)
        y = cp.array(y)
    Xy = xgb.QuantileDMatrix(X, y, max_bin=max_bin, enable_categorical=True)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_categories * n_features, indptr, data, X.dtypes)
    Xy = xgb.DMatrix(X, y, enable_categorical=True)
    xgb.train({'tree_method': tree_method, 'max_bin': max_bin}, Xy)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_categories * n_features, indptr, data, X.dtypes)
    X, y = tm.make_categorical(n_samples, n_features, n_categories, False, sparsity=0.8, cat_ratio=0.5)
    n_cat_features = len([0 for dtype in X.dtypes if is_categorical_dtype(dtype)])
    n_num_features = n_features - n_cat_features
    n_entries = n_categories * n_cat_features + (max_bin + 1) * n_num_features
    Xy = xgb.QuantileDMatrix(X, y, max_bin=max_bin, enable_categorical=True)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_entries, indptr, data, X.dtypes)
    Xy = xgb.DMatrix(X, y, enable_categorical=True)
    xgb.train({'tree_method': tree_method, 'max_bin': max_bin}, Xy)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_entries, indptr, data, X.dtypes)