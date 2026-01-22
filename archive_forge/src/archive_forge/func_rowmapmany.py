from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
def rowmapmany(table, rowgenerator, header, failonerror=None):
    """
    Map each input row to any number of output rows via an arbitrary
    function. E.g.::

        >>> import petl as etl
        >>> table1 = [['id', 'sex', 'age', 'height', 'weight'],
        ...           [1, 'male', 16, 1.45, 62.0],
        ...           [2, 'female', 19, 1.34, 55.4],
        ...           [3, '-', 17, 1.78, 74.4],
        ...           [4, 'male', 21, 1.33]]
        >>> def rowgenerator(row):
        ...     transmf = {'male': 'M', 'female': 'F'}
        ...     yield [row[0], 'gender',
        ...            transmf[row['sex']] if row['sex'] in transmf else None]
        ...     yield [row[0], 'age_months', row.age * 12]
        ...     yield [row[0], 'bmi', row.height / row.weight ** 2]
        ...
        >>> table2 = etl.rowmapmany(table1, rowgenerator,
        ...                         header=['subject_id', 'variable', 'value'])
        >>> table2.lookall()
        +------------+--------------+-----------------------+
        | subject_id | variable     | value                 |
        +============+==============+=======================+
        |          1 | 'gender'     | 'M'                   |
        +------------+--------------+-----------------------+
        |          1 | 'age_months' |                   192 |
        +------------+--------------+-----------------------+
        |          1 | 'bmi'        | 0.0003772112382934443 |
        +------------+--------------+-----------------------+
        |          2 | 'gender'     | 'F'                   |
        +------------+--------------+-----------------------+
        |          2 | 'age_months' |                   228 |
        +------------+--------------+-----------------------+
        |          2 | 'bmi'        | 0.0004366015456998006 |
        +------------+--------------+-----------------------+
        |          3 | 'gender'     | None                  |
        +------------+--------------+-----------------------+
        |          3 | 'age_months' |                   204 |
        +------------+--------------+-----------------------+
        |          3 | 'bmi'        | 0.0003215689675106949 |
        +------------+--------------+-----------------------+
        |          4 | 'gender'     | 'M'                   |
        +------------+--------------+-----------------------+
        |          4 | 'age_months' |                   252 |
        +------------+--------------+-----------------------+

    The `rowgenerator` function should accept a single row and yield zero or
    more rows (lists or tuples).

    The `failonerror` keyword argument is documented under
    :func:`petl.config.failonerror`

    See also the :func:`petl.transform.reshape.melt` function.

    """
    return RowMapManyView(table, rowgenerator, header, failonerror=failonerror)