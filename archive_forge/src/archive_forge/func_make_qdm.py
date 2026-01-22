from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def make_qdm(data: Dict[str, List[np.ndarray]], dev_ordinal: Optional[int], meta: Dict[str, Any], ref: Optional[DMatrix], params: Dict[str, Any]) -> DMatrix:
    """Handle empty partition for QuantileDMatrix."""
    if not data:
        return QuantileDMatrix(np.empty((0, 0)), ref=ref)
    it = PartIter(data, dev_ordinal, **meta)
    m = QuantileDMatrix(it, **params, ref=ref)
    return m