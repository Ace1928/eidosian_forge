import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def substance_names(self):
    """Returns a tuple of the substances' names"""
    return tuple((substance.name for substance in self.substances.values()))