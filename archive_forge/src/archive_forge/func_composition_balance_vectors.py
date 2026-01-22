import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def composition_balance_vectors(self):
    """Returns a list of lists with compositions and a list of composition keys.

        The list of lists can be viewed as a matrix with rows corresponding to composition keys
        (which are given as the second item in the returned tuple) and columns corresponding to
        substances. Multiplying the matrix with a vector of concentrations give an equation which
        is an invariant (corresponds to mass & charge conservation).

        Examples
        --------
        >>> s = 'Cu+2 + NH3 -> CuNH3+2'
        >>> import re
        >>> substances = re.split(r' \\+ | -> ', s)
        >>> rsys = ReactionSystem.from_string(s, substances)
        >>> rsys.composition_balance_vectors()
        ([[2, 0, 2], [0, 3, 3], [0, 1, 1], [1, 0, 1]], [0, 1, 7, 29])

        Returns
        -------
        A: list of lists
        ck: (sorted) tuple of composition keys

        """
    subs = self.substances.values()
    ck = Substance.composition_keys(subs)
    return ([[s.composition.get(k, 0) for s in subs] for k in ck], ck)