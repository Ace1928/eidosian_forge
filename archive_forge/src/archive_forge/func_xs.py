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
def xs(self, key: IndexLabel, axis: Axis=0, level: IndexLabel | None=None, drop_level: bool_t=True) -> Self:
    """
        Return cross-section from the Series/DataFrame.

        This method takes a `key` argument to select data at a particular
        level of a MultiIndex.

        Parameters
        ----------
        key : label or tuple of label
            Label contained in the index, or partially in a MultiIndex.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis to retrieve cross-section on.
        level : object, defaults to first n levels (n=1 or len(key))
            In case of a key partially contained in a MultiIndex, indicate
            which levels are used. Levels can be referred by label or position.
        drop_level : bool, default True
            If False, returns object with same levels as self.

        Returns
        -------
        Series or DataFrame
            Cross-section from the original Series or DataFrame
            corresponding to the selected index levels.

        See Also
        --------
        DataFrame.loc : Access a group of rows and columns
            by label(s) or a boolean array.
        DataFrame.iloc : Purely integer-location based indexing
            for selection by position.

        Notes
        -----
        `xs` can not be used to set values.

        MultiIndex Slicers is a generic way to get/set values on
        any level or levels.
        It is a superset of `xs` functionality, see
        :ref:`MultiIndex Slicers <advanced.mi_slicers>`.

        Examples
        --------
        >>> d = {'num_legs': [4, 4, 2, 2],
        ...      'num_wings': [0, 0, 2, 2],
        ...      'class': ['mammal', 'mammal', 'mammal', 'bird'],
        ...      'animal': ['cat', 'dog', 'bat', 'penguin'],
        ...      'locomotion': ['walks', 'walks', 'flies', 'walks']}
        >>> df = pd.DataFrame(data=d)
        >>> df = df.set_index(['class', 'animal', 'locomotion'])
        >>> df
                                   num_legs  num_wings
        class  animal  locomotion
        mammal cat     walks              4          0
               dog     walks              4          0
               bat     flies              2          2
        bird   penguin walks              2          2

        Get values at specified index

        >>> df.xs('mammal')
                           num_legs  num_wings
        animal locomotion
        cat    walks              4          0
        dog    walks              4          0
        bat    flies              2          2

        Get values at several indexes

        >>> df.xs(('mammal', 'dog', 'walks'))
        num_legs     4
        num_wings    0
        Name: (mammal, dog, walks), dtype: int64

        Get values at specified index and level

        >>> df.xs('cat', level=1)
                           num_legs  num_wings
        class  locomotion
        mammal walks              4          0

        Get values at several indexes and levels

        >>> df.xs(('bird', 'walks'),
        ...       level=[0, 'locomotion'])
                 num_legs  num_wings
        animal
        penguin         2          2

        Get values at specified column and axis

        >>> df.xs('num_wings', axis=1)
        class   animal   locomotion
        mammal  cat      walks         0
                dog      walks         0
                bat      flies         2
        bird    penguin  walks         2
        Name: num_wings, dtype: int64
        """
    axis = self._get_axis_number(axis)
    labels = self._get_axis(axis)
    if isinstance(key, list):
        raise TypeError('list keys are not supported in xs, pass a tuple instead')
    if level is not None:
        if not isinstance(labels, MultiIndex):
            raise TypeError('Index must be a MultiIndex')
        loc, new_ax = labels.get_loc_level(key, level=level, drop_level=drop_level)
        _indexer = [slice(None)] * self.ndim
        _indexer[axis] = loc
        indexer = tuple(_indexer)
        result = self.iloc[indexer]
        setattr(result, result._get_axis_name(axis), new_ax)
        return result
    if axis == 1:
        if drop_level:
            return self[key]
        index = self.columns
    else:
        index = self.index
    if isinstance(index, MultiIndex):
        loc, new_index = index._get_loc_level(key, level=0)
        if not drop_level:
            if lib.is_integer(loc):
                new_index = index[loc:loc + 1]
            else:
                new_index = index[loc]
    else:
        loc = index.get_loc(key)
        if isinstance(loc, np.ndarray):
            if loc.dtype == np.bool_:
                inds, = loc.nonzero()
                return self._take_with_is_copy(inds, axis=axis)
            else:
                return self._take_with_is_copy(loc, axis=axis)
        if not is_scalar(loc):
            new_index = index[loc]
    if is_scalar(loc) and axis == 0:
        if self.ndim == 1:
            return self._values[loc]
        new_mgr = self._mgr.fast_xs(loc)
        result = self._constructor_sliced_from_mgr(new_mgr, axes=new_mgr.axes)
        result._name = self.index[loc]
        result = result.__finalize__(self)
    elif is_scalar(loc):
        result = self.iloc[:, slice(loc, loc + 1)]
    elif axis == 1:
        result = self.iloc[:, loc]
    else:
        result = self.iloc[loc]
        result.index = new_index
    result._set_is_copy(self, copy=not result._is_view)
    return result