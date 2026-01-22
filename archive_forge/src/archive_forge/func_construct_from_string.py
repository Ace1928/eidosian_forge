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
@classmethod
def construct_from_string(cls, string: str):
    """
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[H]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> import re
        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\\[(?P<arg_name>.+)\\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a '{cls.__name__}' from '{string}'"
        ...         )
        """
    import ast
    import re
    if not isinstance(string, str):
        raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
    regex = '^(TensorDtype|numpy.ndarray)\\(shape=(\\((?:(?:\\d+|None),?\\s?)*\\)), dtype=(\\w+)\\)$'
    m = re.search(regex, string)
    err_msg = f"Cannot construct a '{cls.__name__}' from '{string}'; expected a string like 'TensorDtype(shape=(1, 2, 3), dtype=int64)'."
    if m is None:
        raise TypeError(err_msg)
    groups = m.groups()
    if len(groups) != 3:
        raise TypeError(err_msg)
    _, shape, dtype = groups
    shape = ast.literal_eval(shape)
    dtype = np.dtype(dtype)
    return cls(shape, dtype)