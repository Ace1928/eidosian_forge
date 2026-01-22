import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
Returns a list of index pairs of reactions forming equilibria.

        The pairs are sorted with respect to index (lowest first)
        