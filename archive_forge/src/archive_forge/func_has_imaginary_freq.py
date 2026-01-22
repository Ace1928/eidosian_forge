from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def has_imaginary_freq(self, tol: float=0.01) -> bool:
    """True if imaginary frequencies are present anywhere in the band structure. Always True if
        has_imaginary_gamma_freq is True.

        Args:
            tol: Tolerance for determining if a frequency is imaginary. Defaults to 0.01.
        """
    return self.min_freq()[1] + tol < 0