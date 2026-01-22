import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def is_pd_sparse_dtype(dtype: PandasDType) -> bool:
    """Wrapper for testing pandas sparse type."""
    import pandas as pd
    if hasattr(pd.util, 'version') and hasattr(pd.util.version, 'Version'):
        Version = pd.util.version.Version
        if Version(pd.__version__) >= Version('2.1.0'):
            from pandas import SparseDtype
            return isinstance(dtype, SparseDtype)
    from pandas.api.types import is_sparse
    return is_sparse(dtype)