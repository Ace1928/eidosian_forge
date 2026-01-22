import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def enumerate_lists(lists, n=0, filter=lambda x: True):
    ans = []
    for L in lists:
        ans.append([n + i for i, x in enumerate(L) if filter(n + i)])
        n += len(L)
    return ans