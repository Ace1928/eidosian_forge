from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def create_dmatrix_from_partitions(iterator: Iterator[pd.DataFrame], feature_cols: Optional[Sequence[str]], dev_ordinal: Optional[int], use_qdm: bool, kwargs: Dict[str, Any], enable_sparse_data_optim: bool, has_validation_col: bool) -> Tuple[DMatrix, Optional[DMatrix]]:
    """Create DMatrix from spark data partitions.

    Parameters
    ----------
    iterator :
        Pyspark partition iterator.
    feature_cols:
        A sequence of feature names, used only when rapids plugin is enabled.
    dev_ordinal:
        Device ordinal, used when GPU is enabled.
    use_qdm :
        Whether QuantileDMatrix should be used instead of DMatrix.
    kwargs :
        Metainfo for DMatrix.
    enable_sparse_data_optim :
        Whether sparse data should be unwrapped
    has_validation:
        Whether there's validation data.

    Returns
    -------
    Training DMatrix and an optional validation DMatrix.
    """
    train_data: Dict[str, List[np.ndarray]] = defaultdict(list)
    valid_data: Dict[str, List[np.ndarray]] = defaultdict(list)
    n_features: int = 0

    def append_m(part: pd.DataFrame, name: str, is_valid: bool) -> None:
        nonlocal n_features
        if name == alias.data or name in part.columns:
            if name == alias.data and feature_cols is not None and (part[feature_cols].shape[0] > 0):
                array: Optional[np.ndarray] = part[feature_cols]
            elif part[name].shape[0] > 0:
                array = part[name]
                if name == alias.data:
                    array = stack_series(array)
            else:
                array = None
            if name == alias.data and array is not None:
                if n_features == 0:
                    n_features = array.shape[1]
                assert n_features == array.shape[1]
            if array is None:
                return
            if is_valid:
                valid_data[name].append(array)
            else:
                train_data[name].append(array)

    def append_m_sparse(part: pd.DataFrame, name: str, is_valid: bool) -> None:
        nonlocal n_features
        if name == alias.data or name in part.columns:
            if name == alias.data:
                array = _read_csr_matrix_from_unwrapped_spark_vec(part)
                if n_features == 0:
                    n_features = array.shape[1]
                assert n_features == array.shape[1]
            else:
                array = part[name]
            if is_valid:
                valid_data[name].append(array)
            else:
                train_data[name].append(array)

    def make(values: Dict[str, List[np.ndarray]], kwargs: Dict[str, Any]) -> DMatrix:
        if len(values) == 0:
            get_logger('XGBoostPySpark').warning('Detected an empty partition in the training data. Consider to enable repartition_random_shuffle')
            return DMatrix(data=np.empty((0, 0)), **kwargs)
        data = concat_or_none(values[alias.data])
        label = concat_or_none(values.get(alias.label, None))
        weight = concat_or_none(values.get(alias.weight, None))
        margin = concat_or_none(values.get(alias.margin, None))
        qid = concat_or_none(values.get(alias.qid, None))
        return DMatrix(data=data, label=label, weight=weight, base_margin=margin, qid=qid, **kwargs)
    if enable_sparse_data_optim:
        append_fn = append_m_sparse
        assert 'missing' in kwargs and kwargs['missing'] == 0.0
    else:
        append_fn = append_m

    def split_params() -> Tuple[Dict[str, Any], Dict[str, Union[int, float, bool]]]:
        non_data_keys = ('max_bin', 'missing', 'silent', 'nthread', 'enable_categorical')
        non_data_params = {}
        meta = {}
        for k, v in kwargs.items():
            if k in non_data_keys:
                non_data_params[k] = v
            else:
                meta[k] = v
        return (meta, non_data_params)
    meta, params = split_params()
    if feature_cols is not None and use_qdm:
        cache_partitions(iterator, append_fn)
        dtrain: DMatrix = make_qdm(train_data, dev_ordinal, meta, None, params)
    elif feature_cols is not None and (not use_qdm):
        cache_partitions(iterator, append_fn)
        dtrain = make(train_data, kwargs)
    elif feature_cols is None and use_qdm:
        cache_partitions(iterator, append_fn)
        dtrain = make_qdm(train_data, dev_ordinal, meta, None, params)
    else:
        cache_partitions(iterator, append_fn)
        dtrain = make(train_data, kwargs)
    if has_validation_col:
        if use_qdm:
            dvalid: Optional[DMatrix] = make_qdm(valid_data, dev_ordinal, meta, dtrain, params)
        else:
            dvalid = make(valid_data, kwargs) if has_validation_col else None
    else:
        dvalid = None
    if dvalid is not None:
        assert dvalid.num_col() == dtrain.num_col()
    return (dtrain, dvalid)