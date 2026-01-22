from __future__ import annotations
from collections.abc import (
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import (
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import (
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import (
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime
import pandas.io.formats.format as fmt
from pandas.io.formats.info import (
import pandas.plotting
@doc(_shared_docs['compare'], dedent('\n        Returns\n        -------\n        Series or DataFrame\n            If axis is 0 or \'index\' the result will be a Series.\n            The resulting index will be a MultiIndex with \'self\' and \'other\'\n            stacked alternately at the inner level.\n\n            If axis is 1 or \'columns\' the result will be a DataFrame.\n            It will have two columns namely \'self\' and \'other\'.\n\n        See Also\n        --------\n        DataFrame.compare : Compare with another DataFrame and show differences.\n\n        Notes\n        -----\n        Matching NaNs will not appear as a difference.\n\n        Examples\n        --------\n        >>> s1 = pd.Series(["a", "b", "c", "d", "e"])\n        >>> s2 = pd.Series(["a", "a", "c", "b", "e"])\n\n        Align the differences on columns\n\n        >>> s1.compare(s2)\n          self other\n        1    b     a\n        3    d     b\n\n        Stack the differences on indices\n\n        >>> s1.compare(s2, align_axis=0)\n        1  self     b\n           other    a\n        3  self     d\n           other    b\n        dtype: object\n\n        Keep all original rows\n\n        >>> s1.compare(s2, keep_shape=True)\n          self other\n        0  NaN   NaN\n        1    b     a\n        2  NaN   NaN\n        3    d     b\n        4  NaN   NaN\n\n        Keep all original rows and also all original values\n\n        >>> s1.compare(s2, keep_shape=True, keep_equal=True)\n          self other\n        0    a     a\n        1    b     a\n        2    c     c\n        3    d     b\n        4    e     e\n        '), klass=_shared_doc_kwargs['klass'])
def compare(self, other: Series, align_axis: Axis=1, keep_shape: bool=False, keep_equal: bool=False, result_names: Suffixes=('self', 'other')) -> DataFrame | Series:
    return super().compare(other=other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal, result_names=result_names)