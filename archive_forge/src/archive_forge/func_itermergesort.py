from __future__ import absolute_import, print_function, division
import os
import heapq
from tempfile import NamedTemporaryFile
import itertools
import logging
from collections import namedtuple
import operator
from petl.compat import pickle, next, text_type
import petl.config as config
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, asindices
def itermergesort(sources, key, header, missing, reverse):
    its = [iter(t) for t in sources]
    src_hdrs = []
    for it in its:
        try:
            src_hdrs.append(next(it))
        except StopIteration:
            src_hdrs.append([])
    if header is None:
        outhdr = list()
        for hdr in src_hdrs:
            for f in list(map(text_type, hdr)):
                if f not in outhdr:
                    outhdr.append(f)
    else:
        outhdr = header
    yield tuple(outhdr)

    def _standardisedata(it, hdr, ofs):
        flds = list(map(text_type, hdr))
        for _row in it:
            try:
                yield tuple((_row[flds.index(fo)] if fo in flds else missing for fo in ofs))
            except IndexError:
                outrow = [missing] * len(ofs)
                for i, fi in enumerate(flds):
                    try:
                        outrow[ofs.index(fi)] = _row[i]
                    except IndexError:
                        pass
                yield tuple(outrow)
    sits = [_standardisedata(it, hdr, outhdr) for hdr, it in zip(src_hdrs, its)]
    getkey = None
    if key is not None:
        indices = asindices(outhdr, key)
        getkey = comparable_itemgetter(*indices)
    for row in _shortlistmergesorted(getkey, reverse, *sits):
        yield row