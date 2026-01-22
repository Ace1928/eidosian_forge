from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
def bool(self) -> bool_t:
    """
        Return the bool of a single element Series or DataFrame.

        .. deprecated:: 2.1.0

           bool is deprecated and will be removed in future version of pandas.
           For ``Series`` use ``pandas.Series.item``.

        This must be a boolean scalar value, either True or False. It will raise a
        ValueError if the Series or DataFrame does not have exactly 1 element, or that
        element is not boolean (integer values 0 and 1 will also raise an exception).

        Returns
        -------
        bool
            The value in the Series or DataFrame.

        See Also
        --------
        Series.astype : Change the data type of a Series, including to boolean.
        DataFrame.astype : Change the data type of a DataFrame, including to boolean.
        numpy.bool_ : NumPy boolean data type, used by pandas for boolean values.

        Examples
        --------
        The method will only work for single element objects with a boolean value:

        >>> pd.Series([True]).bool()  # doctest: +SKIP
        True
        >>> pd.Series([False]).bool()  # doctest: +SKIP
        False

        >>> pd.DataFrame({'col': [True]}).bool()  # doctest: +SKIP
        True
        >>> pd.DataFrame({'col': [False]}).bool()  # doctest: +SKIP
        False

        This is an alternative method and will only work
        for single element objects with a boolean value:

        >>> pd.Series([True]).item()  # doctest: +SKIP
        True
        >>> pd.Series([False]).item()  # doctest: +SKIP
        False
        """
    warnings.warn(f'{type(self).__name__}.bool is now deprecated and will be removed in future version of pandas', FutureWarning, stacklevel=find_stack_level())
    v = self.squeeze()
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    elif is_scalar(v):
        raise ValueError(f'bool cannot act on a non-boolean single element {type(self).__name__}')
    self.__nonzero__()
    return True