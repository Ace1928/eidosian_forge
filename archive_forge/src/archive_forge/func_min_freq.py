from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def min_freq(self) -> tuple[Kpoint, float]:
    """Returns the q-point where the minimum frequency is reached and its value."""
    idx = np.unravel_index(np.argmin(self.bands), self.bands.shape)
    return (self.qpoints[idx[1]], self.bands[idx])