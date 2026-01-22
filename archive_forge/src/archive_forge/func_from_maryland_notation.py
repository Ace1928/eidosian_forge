from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core import Site, Species
from pymatgen.core.tensors import SquareTensor
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
@classmethod
def from_maryland_notation(cls, sigma_iso, omega, kappa) -> Self:
    """
        Initialize from Maryland notation.

        Args:
            sigma_iso ():
            omega ():
            kappa ():

        Returns:
            ChemicalShielding
        """
    sigma_22 = sigma_iso + kappa * omega / 3
    sigma_11 = (3 * sigma_iso - omega - sigma_22) / 2
    sigma_33 = 3 * sigma_iso - sigma_22 - sigma_11
    return cls(np.diag([sigma_11, sigma_22, sigma_33]))