import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_drop_fields_dispatcher)
def drop_fields(base, drop_names, usemask=True, asrecarray=False):
    """
    Return a new array with fields in `drop_names` dropped.

    Nested fields are supported.

    .. versionchanged:: 1.18.0
        `drop_fields` returns an array with 0 fields if all fields are dropped,
        rather than returning ``None`` as it did previously.

    Parameters
    ----------
    base : array
        Input array
    drop_names : string or sequence
        String or sequence of strings corresponding to the names of the
        fields to drop.
    usemask : {False, True}, optional
        Whether to return a masked array or not.
    asrecarray : string or sequence, optional
        Whether to return a recarray or a mrecarray (`asrecarray=True`) or
        a plain ndarray or masked array with flexible dtype. The default
        is False.

    Examples
    --------
    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
    ...   dtype=[('a', np.int64), ('b', [('ba', np.double), ('bb', np.int64)])])
    >>> rfn.drop_fields(a, 'a')
    array([((2., 3),), ((5., 6),)],
          dtype=[('b', [('ba', '<f8'), ('bb', '<i8')])])
    >>> rfn.drop_fields(a, 'ba')
    array([(1, (3,)), (4, (6,))], dtype=[('a', '<i8'), ('b', [('bb', '<i8')])])
    >>> rfn.drop_fields(a, ['ba', 'bb'])
    array([(1,), (4,)], dtype=[('a', '<i8')])
    """
    if _is_string_like(drop_names):
        drop_names = [drop_names]
    else:
        drop_names = set(drop_names)

    def _drop_descr(ndtype, drop_names):
        names = ndtype.names
        newdtype = []
        for name in names:
            current = ndtype[name]
            if name in drop_names:
                continue
            if current.names is not None:
                descr = _drop_descr(current, drop_names)
                if descr:
                    newdtype.append((name, descr))
            else:
                newdtype.append((name, current))
        return newdtype
    newdtype = _drop_descr(base.dtype, drop_names)
    output = np.empty(base.shape, dtype=newdtype)
    output = recursive_fill_fields(base, output)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)