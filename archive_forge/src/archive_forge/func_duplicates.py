from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, itervalues
from petl.transform.sorts import sort
def duplicates(table, key=None, presorted=False, buffersize=None, tempdir=None, cache=True):
    """
    Select rows with duplicate values under a given key (or duplicate
    rows where no key is given). E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['A', 1, 2.0],
        ...           ['B', 2, 3.4],
        ...           ['D', 6, 9.3],
        ...           ['B', 3, 7.8],
        ...           ['B', 2, 12.3],
        ...           ['E', None, 1.3],
        ...           ['D', 4, 14.5]]
        >>> table2 = etl.duplicates(table1, 'foo')
        >>> table2
        +-----+-----+------+
        | foo | bar | baz  |
        +=====+=====+======+
        | 'B' |   2 |  3.4 |
        +-----+-----+------+
        | 'B' |   3 |  7.8 |
        +-----+-----+------+
        | 'B' |   2 | 12.3 |
        +-----+-----+------+
        | 'D' |   6 |  9.3 |
        +-----+-----+------+
        | 'D' |   4 | 14.5 |
        +-----+-----+------+

        >>> # compound keys are supported
        ... table3 = etl.duplicates(table1, key=['foo', 'bar'])
        >>> table3
        +-----+-----+------+
        | foo | bar | baz  |
        +=====+=====+======+
        | 'B' |   2 |  3.4 |
        +-----+-----+------+
        | 'B' |   2 | 12.3 |
        +-----+-----+------+
        
    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are 
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.
    
    See also :func:`petl.transform.dedup.unique` and
    :func:`petl.transform.dedup.distinct`.
    
    """
    return DuplicatesView(table, key=key, presorted=presorted, buffersize=buffersize, tempdir=tempdir, cache=cache)