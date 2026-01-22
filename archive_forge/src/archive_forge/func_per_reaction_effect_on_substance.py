import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def per_reaction_effect_on_substance(self, substance_key):
    result = {}
    for ri, rxn in enumerate(self.rxns):
        n, = rxn.net_stoich((substance_key,))
        if n != 0:
            result[ri] = n
    return result