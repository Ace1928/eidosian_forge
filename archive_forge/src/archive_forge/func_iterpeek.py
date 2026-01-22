from __future__ import absolute_import, print_function, division
import re
from itertools import islice, chain, cycle, product,\
import operator
from collections import Counter, namedtuple, OrderedDict
from itertools import compress, combinations_with_replacement
from petl.compat import imap, izip, izip_longest, ifilter, ifilterfalse, \
from petl.errors import FieldSelectionError
from petl.comparison import comparable_itemgetter
def iterpeek(it, n=1):
    it = iter(it)
    if n == 1:
        peek = next(it)
        return (peek, chain([peek], it))
    else:
        peek = list(islice(it, n))
        return (peek, chain(peek, it))