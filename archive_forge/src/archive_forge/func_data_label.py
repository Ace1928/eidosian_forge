from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
@property
def data_label(self) -> str:
    """
        Return data label of Stata file.

        Examples
        --------
        >>> df = pd.DataFrame([(1,)], columns=["variable"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> data_label = "This is a data file."
        >>> path = "/My_path/filename.dta"
        >>> df.to_stata(path, time_stamp=time_stamp,    # doctest: +SKIP
        ...             data_label=data_label,  # doctest: +SKIP
        ...             version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.data_label)  # doctest: +SKIP
        This is a data file.
        """
    self._ensure_open()
    return self._data_label