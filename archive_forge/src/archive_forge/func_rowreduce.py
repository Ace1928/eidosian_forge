from __future__ import absolute_import, print_function, division
import itertools
import operator
from collections import OrderedDict
from petl.compat import next, string_types, reduce, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, iterpeek, rowgroupby
from petl.util.base import values
from petl.util.counting import nrows
from petl.transform.sorts import sort, mergesort
from petl.transform.basics import cut
from petl.transform.dedup import distinct
def rowreduce(table, key, reducer, header=None, presorted=False, buffersize=None, tempdir=None, cache=True):
    """
    Group rows under the given key then apply `reducer` to produce a single
    output row for each input group of rows. E.g.::
    
        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 3],
        ...           ['a', 7],
        ...           ['b', 2],
        ...           ['b', 1],
        ...           ['b', 9],
        ...           ['c', 4]]
        >>> def sumbar(key, rows):
        ...     return [key, sum(row[1] for row in rows)]
        ...
        >>> table2 = etl.rowreduce(table1, key='foo', reducer=sumbar,
        ...                        header=['foo', 'barsum'])
        >>> table2
        +-----+--------+
        | foo | barsum |
        +=====+========+
        | 'a' |     10 |
        +-----+--------+
        | 'b' |     12 |
        +-----+--------+
        | 'c' |      4 |
        +-----+--------+
    
    N.B., this is not strictly a "reduce" in the sense of the standard Python
    :func:`reduce` function, i.e., the `reducer` function is *not* applied 
    recursively to values within a group, rather it is applied once to each row 
    group as a whole.
    
    See also :func:`petl.transform.reductions.aggregate` and
    :func:`petl.transform.reductions.fold`.
    
    """
    return RowReduceView(table, key, reducer, header=header, presorted=presorted, buffersize=buffersize, tempdir=tempdir, cache=cache)