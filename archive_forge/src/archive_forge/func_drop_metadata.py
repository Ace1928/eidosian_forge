import os
import sys
import textwrap
import types
import re
import warnings
import functools
import platform
from .._utils import set_module
from numpy.core.numerictypes import issubclass_, issubsctype, issubdtype
from numpy.core import ndarray, ufunc, asarray
import numpy as np
def drop_metadata(dtype, /):
    """
    Returns the dtype unchanged if it contained no metadata or a copy of the
    dtype if it (or any of its structure dtypes) contained metadata.

    This utility is used by `np.save` and `np.savez` to drop metadata before
    saving.

    .. note::

        Due to its limitation this function may move to a more appropriate
        home or change in the future and is considered semi-public API only.

    .. warning::

        This function does not preserve more strange things like record dtypes
        and user dtypes may simply return the wrong thing.  If you need to be
        sure about the latter, check the result with:
        ``np.can_cast(new_dtype, dtype, casting="no")``.

    """
    if dtype.fields is not None:
        found_metadata = dtype.metadata is not None
        names = []
        formats = []
        offsets = []
        titles = []
        for name, field in dtype.fields.items():
            field_dt = drop_metadata(field[0])
            if field_dt is not field[0]:
                found_metadata = True
            names.append(name)
            formats.append(field_dt)
            offsets.append(field[1])
            titles.append(None if len(field) < 3 else field[2])
        if not found_metadata:
            return dtype
        structure = dict(names=names, formats=formats, offsets=offsets, titles=titles, itemsize=dtype.itemsize)
        return np.dtype(structure, align=dtype.isalignedstruct)
    elif dtype.subdtype is not None:
        subdtype, shape = dtype.subdtype
        new_subdtype = drop_metadata(subdtype)
        if dtype.metadata is None and new_subdtype is subdtype:
            return dtype
        return np.dtype((new_subdtype, shape))
    else:
        if dtype.metadata is None:
            return dtype
        return np.dtype(dtype.str)