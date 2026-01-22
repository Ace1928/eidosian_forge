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
def joinrows(_lrowgrp, _rrowgrp):
    if _rrowgrp is None:
        for lrow in _lrowgrp:
            outrow = list(lrow)
            outrow.extend([missing] * len(rvind))
            yield tuple(outrow)
    else:
        rrow = next(iter(_rrowgrp))
        for lrow in _lrowgrp:
            outrow = list(lrow)
            outrow.extend(rgetv(rrow))
            yield tuple(outrow)