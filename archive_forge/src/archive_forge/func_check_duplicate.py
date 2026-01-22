import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def check_duplicate(self, throw=False):
    """Raies ValueError if there are duplicates in ``self.rxns``"""
    for i1, rxn1 in enumerate(self.rxns):
        for i2, rxn2 in enumerate(self.rxns[i1 + 1:], i1 + 1):
            if rxn1 == rxn2:
                if throw:
                    raise ValueError('Duplicate reactions %d & %d: %s' % (i1, i2, rxn1.string(with_param=False, with_name=False)))
                else:
                    return False
    return True