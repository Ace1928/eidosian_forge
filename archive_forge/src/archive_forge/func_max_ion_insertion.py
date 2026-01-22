from __future__ import annotations
import math
from collections import defaultdict
import scipy.constants as const
from pymatgen.core import Composition, Element, Species
@property
def max_ion_insertion(self):
    """Maximum number of ion A that can be inserted while maintaining charge-balance.
        No consideration is given to whether there (geometrically speaking) are ion sites to actually accommodate the
        extra ions.

        Returns:
            integer amount of ion. Depends on cell size (this is an 'extrinsic' function!)
        """
    if self.working_ion_charge < 0:
        pot_sum = sum(((spec.oxi_state - Element(spec.symbol).max_oxidation_state) * self.comp[spec] for spec in self.comp if is_redox_active_intercalation(Element(spec.symbol))))
    else:
        lowest_oxid = defaultdict(lambda: 2, {'Cu': 1})
        pot_sum = sum(((spec.oxi_state - min((os for os in Element(spec.symbol).oxidation_states if os >= lowest_oxid[spec.symbol]))) * self.comp[spec] for spec in self.comp if is_redox_active_intercalation(Element(spec.symbol))))
    return pot_sum / self.working_ion_charge