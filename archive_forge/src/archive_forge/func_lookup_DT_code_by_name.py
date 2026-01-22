import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def lookup_DT_code_by_name(name):
    """
    >>> lookup_DT_code_by_name('K12n123')
    'lalbdFaihCjlkge.001101000111'
    >>> lookup_DT_code_by_name('8_20')
    'hahDeHFgCaB.01010001'
    >>> lookup_DT_code_by_name('8a1')
    'hahbdegahcf.01000011'
    >>> lookup_DT_code_by_name('garbage')
    """
    if re.match('\\d+[an]\\d+$', name):
        name = 'K' + name
    for table in DT_tables:
        try:
            return str(table[name])
        except IndexError:
            continue