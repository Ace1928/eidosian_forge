from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def stringpatterncounter(table, field):
    """
    Profile string patterns in the given field, returning a :class:`dict`
    mapping patterns to counts.

    """
    trans = maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', 'AAAAAAAAAAAAAAAAAAAAAAAAAAaaaaaaaaaaaaaaaaaaaaaaaaaa9999999999')
    counter = Counter()
    for v in values(table, field):
        p = str(v).translate(trans)
        counter[p] += 1
    return counter