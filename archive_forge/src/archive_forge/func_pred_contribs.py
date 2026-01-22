from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def pred_contribs(model: XGBModel, data: ArrayLike, base_margin: Optional[ArrayLike]=None, strict_shape: bool=False) -> np.ndarray:
    """Predict contributions with data with the full model."""
    iteration_range = model._get_iteration_range(None)
    data_dmatrix = DMatrix(data, base_margin=base_margin, missing=model.missing, nthread=model.n_jobs, feature_types=model.feature_types, enable_categorical=model.enable_categorical)
    return model.get_booster().predict(data_dmatrix, pred_contribs=True, validate_features=False, iteration_range=iteration_range, strict_shape=strict_shape)