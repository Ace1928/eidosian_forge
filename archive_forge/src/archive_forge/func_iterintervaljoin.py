from __future__ import absolute_import, print_function, division
from operator import itemgetter, attrgetter
from petl.compat import text_type
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort
from collections import namedtuple
def iterintervaljoin(left, right, lstart, lstop, rstart, rstop, lkey, rkey, include_stop, missing, lprefix, rprefix, leftouter, anti=False):
    lit = iter(left)
    lhdr = next(lit)
    lflds = list(map(text_type, lhdr))
    rit = iter(right)
    rhdr = next(rit)
    rflds = list(map(text_type, rhdr))
    asindices(lhdr, lstart)
    asindices(lhdr, lstop)
    if lkey is not None:
        asindices(lhdr, lkey)
    asindices(rhdr, rstart)
    asindices(rhdr, rstop)
    if rkey is not None:
        asindices(rhdr, rkey)
    if lprefix is None:
        outhdr = list(lflds)
        if not anti:
            outhdr.extend(rflds)
    else:
        outhdr = list((lprefix + f for f in lflds))
        if not anti:
            outhdr.extend((rprefix + f for f in rflds))
    yield tuple(outhdr)
    getlstart = itemgetter(lflds.index(lstart))
    getlstop = itemgetter(lflds.index(lstop))
    if rkey is None:
        lookup = intervallookup(right, rstart, rstop, include_stop=include_stop)
        search = lookup.search
        for lrow in lit:
            start = getlstart(lrow)
            stop = getlstop(lrow)
            rrows = search(start, stop)
            if rrows:
                if not anti:
                    for rrow in rrows:
                        outrow = list(lrow)
                        outrow.extend(rrow)
                        yield tuple(outrow)
            elif leftouter:
                outrow = list(lrow)
                if not anti:
                    outrow.extend([missing] * len(rflds))
                yield tuple(outrow)
    else:
        lookup = facetintervallookup(right, key=rkey, start=rstart, stop=rstop, include_stop=include_stop)
        search = dict()
        for f in lookup:
            search[f] = lookup[f].search
        getlkey = itemgetter(*asindices(lflds, lkey))
        for lrow in lit:
            lkey = getlkey(lrow)
            start = getlstart(lrow)
            stop = getlstop(lrow)
            try:
                rrows = search[lkey](start, stop)
            except KeyError:
                rrows = None
            except AttributeError:
                rrows = None
            if rrows:
                if not anti:
                    for rrow in rrows:
                        outrow = list(lrow)
                        outrow.extend(rrow)
                        yield tuple(outrow)
            elif leftouter:
                outrow = list(lrow)
                if not anti:
                    outrow.extend([missing] * len(rflds))
                yield tuple(outrow)