from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
def recordcomplement(a, b, buffersize=None, tempdir=None, cache=True, strict=False):
    """
    Find records in `a` that are not in `b`. E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['bar', 'foo', 'baz'],
        ...      [2, 'B', False],
        ...      [9, 'A', False],
        ...      [3, 'B', True],
        ...      [9, 'C', True]]
        >>> aminusb = etl.recordcomplement(a, b)
        >>> aminusb
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

        >>> bminusa = etl.recordcomplement(b, a)
        >>> bminusa
        +-----+-----+-------+
        | bar | foo | baz   |
        +=====+=====+=======+
        |   3 | 'B' | True  |
        +-----+-----+-------+
        |   9 | 'A' | False |
        +-----+-----+-------+

    Note that both tables must have the same set of fields, but that the order
    of the fields does not matter. See also the
    :func:`petl.transform.setops.complement` function.

    See also the discussion of the `buffersize`, `tempdir` and `cache` arguments
    under the :func:`petl.transform.sorts.sort` function.

    """
    ha = header(a)
    hb = header(b)
    assert set(ha) == set(hb), 'both tables must have the same set of fields'
    bv = cut(b, *ha)
    return complement(a, bv, buffersize=buffersize, tempdir=tempdir, cache=cache, strict=strict)