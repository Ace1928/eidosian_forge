import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def propagate_phi_map(phismap):
    """An iterative dataflow algorithm to find the definition
            (the source) of each PHI node.
            """
    blacklist = defaultdict(set)
    while True:
        changing = False
        for phi, defsites in sorted(list(phismap.items())):
            for rhs, state in sorted(list(defsites)):
                if rhs in phi_set:
                    defsites |= phismap[rhs]
                    blacklist[phi].add((rhs, state))
            to_remove = blacklist[phi]
            if to_remove & defsites:
                defsites -= to_remove
                changing = True
        _logger.debug('changing phismap: %s', _lazy_pformat(phismap))
        if not changing:
            break