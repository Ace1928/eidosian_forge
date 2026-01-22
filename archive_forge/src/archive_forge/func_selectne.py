from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, string_types, callable, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError
from petl.util.base import asindices, expr, Table, values, Record
def selectne(table, field, value, complement=False):
    """Select rows where the given field does not equal the given value."""
    return selectop(table, field, value, operator.ne, complement=complement)