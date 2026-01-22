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
def groupselectfirst(table, key, presorted=False, buffersize=None, tempdir=None, cache=True):
    """Group by the `key` field then return the first row within each group.
    E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['A', 1, True],
        ...           ['C', 7, False],
        ...           ['B', 2, False],
        ...           ['C', 9, True]]
        >>> table2 = etl.groupselectfirst(table1, key='foo')
        >>> table2
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'B' |   2 | False |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

    See also :func:`petl.transform.reductions.groupselectlast`,
    :func:`petl.transform.dedup.distinct`.

    """

    def _reducer(k, rows):
        return next(rows)
    return rowreduce(table, key, reducer=_reducer, presorted=presorted, buffersize=buffersize, tempdir=tempdir, cache=cache)