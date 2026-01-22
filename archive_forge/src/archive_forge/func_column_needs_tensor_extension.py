import numbers
import os
from packaging.version import Version
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._typing import Dtype
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.indexers import check_array_indexer, validate_indices
from pandas.io.formats.format import ExtensionArrayFormatter
from ray.air.util.tensor_extensions.utils import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def column_needs_tensor_extension(s: pd.Series) -> bool:
    """Return whether the provided pandas Series column needs a tensor extension
    representation. This tensor extension representation provides more efficient slicing
    and interop with ML frameworks.

    Args:
        s: The pandas Series column that may need to be represented using the tensor
            extension.

    Returns:
        Whether the provided Series needs a tensor extension representation.
    """
    return s.dtype.type is np.object_ and (not s.empty) and isinstance(s.iloc[0], np.ndarray)