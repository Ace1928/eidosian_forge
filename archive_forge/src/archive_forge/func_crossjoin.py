from __future__ import absolute_import, print_function, division
import itertools
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.comparison import comparable_itemgetter, Comparable
from petl.util.base import Table, asindices, rowgetter, rowgroupby, \
from petl.transform.sorts import sort
from petl.transform.basics import cut, cutout
from petl.transform.dedup import distinct
def crossjoin(*tables, **kwargs):
    """
    Form the cartesian product of the given tables. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'colour'],
        ...           [1, 'blue'],
        ...           [2, 'red']]
        >>> table2 = [['id', 'shape'],
        ...           [1, 'circle'],
        ...           [3, 'square']]
        >>> table3 = etl.crossjoin(table1, table2)
        >>> table3
        +----+--------+----+----------+
        | id | colour | id | shape    |
        +====+========+====+==========+
        |  1 | 'blue' |  1 | 'circle' |
        +----+--------+----+----------+
        |  1 | 'blue' |  3 | 'square' |
        +----+--------+----+----------+
        |  2 | 'red'  |  1 | 'circle' |
        +----+--------+----+----------+
        |  2 | 'red'  |  3 | 'square' |
        +----+--------+----+----------+

    If `prefix` is `True` then field names in the output table header will be
    prefixed by the index of the input table.

    """
    return CrossJoinView(*tables, **kwargs)