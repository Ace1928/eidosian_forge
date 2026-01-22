from __future__ import absolute_import, print_function, division
from itertools import islice, chain
from collections import deque
from itertools import count
from petl.compat import izip, izip_longest, next, string_types, text_type
from petl.util.base import asindices, rowgetter, Record, Table
import logging
def itercat(sources, missing, header):
    its = [iter(t) for t in sources]
    hdrs = []
    for it in its:
        try:
            hdrs.append(list(next(it)))
        except StopIteration:
            hdrs.append([])
    if header is None:
        outhdr = list(hdrs[0])
        for hdr in hdrs[1:]:
            for h in hdr:
                if h not in outhdr:
                    outhdr.append(h)
    else:
        outhdr = header
    yield tuple(outhdr)
    for hdr, it in zip(hdrs, its):
        for row in it:
            outrow = list()
            for h in outhdr:
                val = missing
                try:
                    val = row[hdr.index(h)]
                except IndexError:
                    pass
                except ValueError:
                    pass
                outrow.append(val)
            yield tuple(outrow)