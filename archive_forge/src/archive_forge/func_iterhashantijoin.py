from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, text_type
from petl.util.base import Table, asindices, rowgetter, iterpeek
from petl.util.lookups import lookup, lookupone
from petl.transform.joins import keys_from_args
def iterhashantijoin(left, right, lkey, rkey):
    lit = iter(left)
    rit = iter(right)
    lhdr = next(lit)
    rhdr = next(rit)
    yield tuple(lhdr)
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)
    lgetk = operator.itemgetter(*lkind)
    rgetk = operator.itemgetter(*rkind)
    rkeys = set()
    for rrow in rit:
        rk = rgetk(rrow)
        rkeys.add(rk)
    for lrow in lit:
        lk = lgetk(lrow)
        if lk not in rkeys:
            yield tuple(lrow)