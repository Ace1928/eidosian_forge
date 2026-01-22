import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def substance_participation(self, substance_key):
    """Returns indices of reactions where substance_key occurs

        Parameters
        ----------
        substance_key: str

        Examples
        --------
        >>> rs = ReactionSystem.from_string('2 H2 + O2 -> 2 H2O\\n 2 H2O2 -> 2 H2O + O2')
        >>> rs.substance_participation('H2')
        [0]
        >>> rs.substance_participation('O2')
        [0, 1]
        >>> rs.substance_participation('O3')
        []

        Returns
        -------
        List of indices for self.rxns where `substance_key` participates

        """
    return [ri for ri, rxn in enumerate(self.rxns) if substance_key in rxn.keys()]