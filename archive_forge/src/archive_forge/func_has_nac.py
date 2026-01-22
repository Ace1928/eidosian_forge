from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
@property
def has_nac(self) -> bool:
    """True if nac_frequencies are present (i.e. the band structure has been
        calculated taking into account Born-charge-derived non-analytical corrections at Gamma).
        """
    return len(self.nac_frequencies) > 0