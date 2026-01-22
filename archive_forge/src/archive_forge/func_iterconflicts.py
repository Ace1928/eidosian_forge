from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, itervalues
from petl.transform.sorts import sort
def iterconflicts(source, key, missing, exclude, include):
    if exclude and (not isinstance(exclude, (list, tuple))):
        exclude = (exclude,)
    if include and (not isinstance(include, (list, tuple))):
        include = (include,)
    if include and exclude:
        include = None
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    yield tuple(hdr)
    indices = asindices(hdr, key)
    getkey = operator.itemgetter(*indices)
    previous = None
    previous_yielded = False
    for row in it:
        if previous is None:
            previous = row
        else:
            kprev = getkey(previous)
            kcurr = getkey(row)
            if kprev == kcurr:
                conflict = False
                for x, y, f in zip(previous, row, flds):
                    if exclude and f not in exclude or (include and f in include) or (not exclude and (not include)):
                        if missing not in (x, y) and x != y:
                            conflict = True
                            break
                if conflict:
                    if not previous_yielded:
                        yield tuple(previous)
                        previous_yielded = True
                    yield tuple(row)
            else:
                previous_yielded = False
            previous = row