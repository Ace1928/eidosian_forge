from __future__ import absolute_import, print_function, division
import os
import heapq
from tempfile import NamedTemporaryFile
import itertools
import logging
from collections import namedtuple
import operator
from petl.compat import pickle, next, text_type
import petl.config as config
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, asindices

    Return True if the table is ordered (i.e., sorted) by the given key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['a', 1, True],
        ...           ['b', 3, True],
        ...           ['b', 2]]
        >>> etl.issorted(table1, key='foo')
        True
        >>> etl.issorted(table1, key='bar')
        False
        >>> etl.issorted(table1, key='foo', strict=True)
        False
        >>> etl.issorted(table1, key='foo', reverse=True)
        False

    