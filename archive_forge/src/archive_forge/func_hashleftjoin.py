from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, text_type
from petl.util.base import Table, asindices, rowgetter, iterpeek
from petl.util.lookups import lookup, lookupone
from petl.transform.joins import keys_from_args
def hashleftjoin(left, right, key=None, lkey=None, rkey=None, missing=None, cache=True, lprefix=None, rprefix=None):
    """Alternative implementation of :func:`petl.transform.joins.leftjoin`,
    where the join is executed by constructing an in-memory lookup for the
    right hand table, then iterating over rows from the left hand table.
    
    May be faster and/or more resource efficient where the right table is small
    and the left table is large.
    
    By default data from right hand table is cached to improve performance
    (only available when `key` is given).

    Left and right tables with different key fields can be handled via the
    `lkey` and `rkey` arguments.

    """
    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return HashLeftJoinView(left, right, lkey, rkey, missing=missing, cache=cache, lprefix=lprefix, rprefix=rprefix)