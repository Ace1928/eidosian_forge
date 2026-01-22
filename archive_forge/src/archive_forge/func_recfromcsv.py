import os
import re
import functools
import itertools
import warnings
import weakref
import contextlib
import operator
from operator import itemgetter, index as opindex, methodcaller
from collections.abc import Mapping
import numpy as np
from . import format
from ._datasource import DataSource
from numpy.core import overrides
from numpy.core.multiarray import packbits, unpackbits
from numpy.core._multiarray_umath import _load_from_filelike
from numpy.core.overrides import set_array_function_like_doc, set_module
from ._iotools import (
from numpy.compat import (
def recfromcsv(fname, **kwargs):
    """
    Load ASCII data stored in a comma-separated file.

    The returned array is a record array (if ``usemask=False``, see
    `recarray`) or a masked record array (if ``usemask=True``,
    see `ma.mrecords.MaskedRecords`).

    Parameters
    ----------
    fname, kwargs : For a description of input parameters, see `genfromtxt`.

    See Also
    --------
    numpy.genfromtxt : generic function to load ASCII data.

    Notes
    -----
    By default, `dtype` is None, which means that the data-type of the output
    array will be determined from the data.

    """
    kwargs.setdefault('case_sensitive', 'lower')
    kwargs.setdefault('names', True)
    kwargs.setdefault('delimiter', ',')
    kwargs.setdefault('dtype', None)
    output = genfromtxt(fname, **kwargs)
    usemask = kwargs.get('usemask', False)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output