import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
@property
def nr(self):
    """Number of reactions"""
    return len(self.rxns)