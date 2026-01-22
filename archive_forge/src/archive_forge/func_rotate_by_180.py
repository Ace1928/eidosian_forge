import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def rotate_by_180(self):
    """Effective reverses directions of the strands"""
    self.rotate(2)