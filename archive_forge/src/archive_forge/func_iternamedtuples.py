from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
def iternamedtuples(table, *sliceargs, **kwargs):
    missing = kwargs.get('missing', None)
    name = kwargs.get('name', 'row')
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    nt = namedtuple(name, tuple(flds))
    if sliceargs:
        it = islice(it, *sliceargs)
    for row in it:
        yield asnamedtuple(nt, row, missing)