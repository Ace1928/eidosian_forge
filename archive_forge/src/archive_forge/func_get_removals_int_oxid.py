from __future__ import annotations
import math
from collections import defaultdict
import scipy.constants as const
from pymatgen.core import Composition, Element, Species
def get_removals_int_oxid(self):
    """Returns a set of ion removal steps, e.g. set([1 2 4]) etc. in order to
        produce integer oxidation states of the redox metals.
        If multiple redox metals are present, all combinations of reduction/oxidation are tested.
        Note that having more than 3 redox metals will likely slow down the algorithm.

        Examples:
            LiFePO4 will return [1]
            Li4Fe3Mn1(PO4)4 will return [1, 2, 3, 4])
            Li6V4(PO4)6 will return [4, 6])  *note that this example is not normalized*

        Returns:
            array of integer ion removals. If you double the unit cell, your answers will be twice as large!
        """
    oxid_els = [Element(spec.symbol) for spec in self.comp if is_redox_active_intercalation(spec)]
    num_a = set()
    for oxid_el in oxid_els:
        num_a = num_a | self._get_int_removals_helper(self.comp.copy(), oxid_el, oxid_els, num_a)
    num_working_ion = self.comp[Species(self.working_ion.symbol, self.working_ion_charge)]
    return {num_working_ion - a for a in num_a}