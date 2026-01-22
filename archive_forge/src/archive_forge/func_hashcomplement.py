from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import next
from petl.comparison import Comparable
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut
def hashcomplement(a, b, strict=False):
    """
    Alternative implementation of :func:`petl.transform.setops.complement`,
    where the complement is executed by constructing an in-memory set for all
    rows found in the right hand table, then iterating over rows from the
    left hand table.

    May be faster and/or more resource efficient where the right table is small
    and the left table is large.
    
    .. versionchanged:: 1.1.0
    
    If `strict` is `True` then strict set-like behaviour is used, i.e., 
    only rows in `a` not found in `b` are returned.

    """
    return HashComplementView(a, b, strict=strict)