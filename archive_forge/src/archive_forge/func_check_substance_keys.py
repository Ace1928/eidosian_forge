import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def check_substance_keys(self, throw=False):
    for rxn in self.rxns:
        for key in chain(rxn.reac, rxn.prod, rxn.inact_reac, rxn.inact_prod):
            if key not in self.substances:
                if throw:
                    raise ValueError('Unknown key: %s' % key)
                else:
                    return False
    return True