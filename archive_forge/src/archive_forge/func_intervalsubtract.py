from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def intervalsubtract(left, right, lstart='start', lstop='stop', rstart='start', rstop='stop', lkey=None, rkey=None, include_stop=False):
    """
    Subtract intervals in the right hand table from intervals in the left hand 
    table.
    
    """
    assert (lkey is None) == (rkey is None), 'facet key field must be provided for both or neither table'
    return IntervalSubtractView(left, right, lstart=lstart, lstop=lstop, rstart=rstart, rstop=rstop, lkey=lkey, rkey=rkey, include_stop=include_stop)