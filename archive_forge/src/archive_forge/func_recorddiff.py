from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
def recorddiff(a, b, buffersize=None, tempdir=None, cache=True, strict=False):
    """
    Find the difference between records in two tables. E.g.::

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
        >>> added, subtracted = etl.recorddiff(a, b)
        >>> added
        +-----+-----+-------+
        | bar | foo | baz   |
        +=====+=====+=======+
        |   3 | 'B' | True  |
        +-----+-----+-------+
        |   9 | 'A' | False |
        +-----+-----+-------+

        >>> subtracted
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

    Convenient shorthand for
    ``(recordcomplement(b, a), recordcomplement(a, b))``. See also
    :func:`petl.transform.setops.recordcomplement`.

    See also the discussion of the `buffersize`, `tempdir` and `cache`
    arguments under the :func:`petl.transform.sorts.sort` function.

    .. versionchanged:: 1.1.0

    If `strict` is `True` then strict set-like behaviour is used.

    """
    added = recordcomplement(b, a, buffersize=buffersize, tempdir=tempdir, cache=cache, strict=strict)
    subtracted = recordcomplement(a, b, buffersize=buffersize, tempdir=tempdir, cache=cache, strict=strict)
    return (added, subtracted)