from __future__ import absolute_import, print_function, division
import itertools
import collections
import operator
from petl.compat import next, text_type
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, rowgetter, values, itervalues, \
from petl.transform.sorts import sort
def recast(table, key=None, variablefield='variable', valuefield='value', samplesize=1000, reducers=None, missing=None):
    """
    Recast molten data. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'variable', 'value'],
        ...           [3, 'age', 16],
        ...           [1, 'gender', 'F'],
        ...           [2, 'gender', 'M'],
        ...           [2, 'age', 17],
        ...           [1, 'age', 12],
        ...           [3, 'gender', 'M']]
        >>> table2 = etl.recast(table1)
        >>> table2
        +----+-----+--------+
        | id | age | gender |
        +====+=====+========+
        |  1 |  12 | 'F'    |
        +----+-----+--------+
        |  2 |  17 | 'M'    |
        +----+-----+--------+
        |  3 |  16 | 'M'    |
        +----+-----+--------+

        >>> # specifying variable and value fields
        ... table3 = [['id', 'vars', 'vals'],
        ...           [3, 'age', 16],
        ...           [1, 'gender', 'F'],
        ...           [2, 'gender', 'M'],
        ...           [2, 'age', 17],
        ...           [1, 'age', 12],
        ...           [3, 'gender', 'M']]
        >>> table4 = etl.recast(table3, variablefield='vars', valuefield='vals')
        >>> table4
        +----+-----+--------+
        | id | age | gender |
        +====+=====+========+
        |  1 |  12 | 'F'    |
        +----+-----+--------+
        |  2 |  17 | 'M'    |
        +----+-----+--------+
        |  3 |  16 | 'M'    |
        +----+-----+--------+

        >>> # if there are multiple values for each key/variable pair, and no
        ... # reducers function is provided, then all values will be listed
        ... table6 = [['id', 'time', 'variable', 'value'],
        ...           [1, 11, 'weight', 66.4],
        ...           [1, 14, 'weight', 55.2],
        ...           [2, 12, 'weight', 53.2],
        ...           [2, 16, 'weight', 43.3],
        ...           [3, 12, 'weight', 34.5],
        ...           [3, 17, 'weight', 49.4]]
        >>> table7 = etl.recast(table6, key='id')
        >>> table7
        +----+--------------+
        | id | weight       |
        +====+==============+
        |  1 | [66.4, 55.2] |
        +----+--------------+
        |  2 | [53.2, 43.3] |
        +----+--------------+
        |  3 | [34.5, 49.4] |
        +----+--------------+

        >>> # multiple values can be reduced via an aggregation function
        ... def mean(values):
        ...     return float(sum(values)) / len(values)
        ...
        >>> table8 = etl.recast(table6, key='id', reducers={'weight': mean})
        >>> table8
        +----+--------------------+
        | id | weight             |
        +====+====================+
        |  1 | 60.800000000000004 |
        +----+--------------------+
        |  2 |              48.25 |
        +----+--------------------+
        |  3 |              41.95 |
        +----+--------------------+

        >>> # missing values are padded with whatever is provided via the
        ... # missing keyword argument (None by default)
        ... table9 = [['id', 'variable', 'value'],
        ...           [1, 'gender', 'F'],
        ...           [2, 'age', 17],
        ...           [1, 'age', 12],
        ...           [3, 'gender', 'M']]
        >>> table10 = etl.recast(table9, key='id')
        >>> table10
        +----+------+--------+
        | id | age  | gender |
        +====+======+========+
        |  1 |   12 | 'F'    |
        +----+------+--------+
        |  2 |   17 | None   |
        +----+------+--------+
        |  3 | None | 'M'    |
        +----+------+--------+

    Note that the table is scanned once to discover variables, then a second
    time to reshape the data and recast variables as fields. How many rows are
    scanned in the first pass is determined by the `samplesize` argument.

    See also :func:`petl.transform.reshape.melt`.

    """
    return RecastView(table, key=key, variablefield=variablefield, valuefield=valuefield, samplesize=samplesize, reducers=reducers, missing=missing)