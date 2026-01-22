from __future__ import absolute_import, print_function, division
import re
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, asindices
from petl.transform.basics import TransformError
from petl.transform.conversions import convert
def searchcomplement(table, *args, **kwargs):
    """
    Perform a regular expression search, returning rows that **do not**
    match a given pattern, either anywhere in the row or within a specific
    field. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['orange', 12, 'oranges are nice fruit'],
        ...           ['mango', 42, 'I like them'],
        ...           ['banana', 74, 'lovely too'],
        ...           ['cucumber', 41, 'better than mango']]
        >>> # search any field
        ... table2 = etl.searchcomplement(table1, '.g.')
        >>> table2
        +----------+-----+--------------+
        | foo      | bar | baz          |
        +==========+=====+==============+
        | 'banana' |  74 | 'lovely too' |
        +----------+-----+--------------+

        >>> # search a specific field
        ... table3 = etl.searchcomplement(table1, 'foo', '.g.')
        >>> table3
        +------------+-----+---------------------+
        | foo        | bar | baz                 |
        +============+=====+=====================+
        | 'banana'   |  74 | 'lovely too'        |
        +------------+-----+---------------------+
        | 'cucumber' |  41 | 'better than mango' |
        +------------+-----+---------------------+

    This returns the complement of :func:`petl.transform.regex.search`.

    """
    return search(table, *args, complement=True, **kwargs)