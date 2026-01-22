import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def is_incoming(self, i):
    if self.sign == 1:
        return i in (0, 3)
    elif self.sign == -1:
        return i in (0, 1)
    else:
        raise ValueError('Crossing not oriented')