import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def toflex(self):
    """
        Transforms a masked array into a flexible-type array.

        The flexible type array that is returned will have two fields:

        * the ``_data`` field stores the ``_data`` part of the array.
        * the ``_mask`` field stores the ``_mask`` part of the array.

        Parameters
        ----------
        None

        Returns
        -------
        record : ndarray
            A new flexible-type `ndarray` with two fields: the first element
            containing a value, the second element containing the corresponding
            mask boolean. The returned record shape matches self.shape.

        Notes
        -----
        A side-effect of transforming a masked array into a flexible `ndarray` is
        that meta information (``fill_value``, ...) will be lost.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.toflex()
        array([[(1, False), (2,  True), (3, False)],
               [(4,  True), (5, False), (6,  True)],
               [(7, False), (8,  True), (9, False)]],
              dtype=[('_data', '<i8'), ('_mask', '?')])

        """
    ddtype = self.dtype
    _mask = self._mask
    if _mask is None:
        _mask = make_mask_none(self.shape, ddtype)
    mdtype = self._mask.dtype
    record = np.ndarray(shape=self.shape, dtype=[('_data', ddtype), ('_mask', mdtype)])
    record['_data'] = self._data
    record['_mask'] = self._mask
    return record