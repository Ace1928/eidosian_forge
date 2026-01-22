import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def strand_label(self):
    return self.crossing.strand_labels[self.strand_index]