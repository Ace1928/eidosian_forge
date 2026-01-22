from __future__ import absolute_import, print_function, division
import operator
from petl.compat import next, text_type
from petl.util.base import Table, asindices, rowgetter, iterpeek
from petl.util.lookups import lookup, lookupone
from petl.transform.joins import keys_from_args
def iterhashrightjoin(left, right, lkey, rkey, missing, llookup, lprefix, rprefix):
    lit = iter(left)
    rit = iter(right)
    lhdr = next(lit)
    rhdr = next(rit)
    lkind = asindices(lhdr, lkey)
    rkind = asindices(rhdr, rkey)
    rgetk = operator.itemgetter(*rkind)
    rvind = [i for i in range(len(rhdr)) if i not in rkind]
    rgetv = rowgetter(*rvind)
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [text_type(lprefix) + text_type(f) for f in lhdr]
    if rprefix is None:
        outhdr.extend(rgetv(rhdr))
    else:
        outhdr.extend([text_type(rprefix) + text_type(f) for f in rgetv(rhdr)])
    yield tuple(outhdr)

    def joinrows(_rrow, _lrows):
        for lrow in _lrows:
            _outrow = list(lrow)
            _outrow.extend(rgetv(_rrow))
            yield tuple(_outrow)
    for rrow in rit:
        k = rgetk(rrow)
        if k in llookup:
            lrows = llookup[k]
            for outrow in joinrows(rrow, lrows):
                yield outrow
        else:
            outrow = [missing] * len(lhdr)
            for li, ri in zip(lkind, rkind):
                outrow[li] = rrow[ri]
            outrow.extend(rgetv(rrow))
            yield tuple(outrow)