from __future__ import division, print_function, absolute_import
from petl.compat import next, string_types
from petl.util.base import iterpeek, ValuesView, Table
from petl.util.materialise import columns
def valuestoarray(vals, dtype=None, count=-1, sample=1000):
    """
    Load values from a table column into a `numpy <http://www.numpy.org/>`_
    array, e.g.::

        >>> import petl as etl
        >>> table = [('foo', 'bar', 'baz'),
        ...          ('apples', 1, 2.5),
        ...          ('oranges', 3, 4.4),
        ...          ('pears', 7, .1)]
        >>> table = etl.wrap(table)
        >>> table.values('bar').array()
        array([1, 3, 7])
        >>> # specify dtype
        ... table.values('bar').array(dtype='i4')
        array([1, 3, 7], dtype=int32)

    """
    import numpy as np
    it = iter(vals)
    if dtype is None:
        peek, it = iterpeek(it, sample)
        dtype = np.array(peek).dtype
    a = np.fromiter(it, dtype=dtype, count=count)
    return a