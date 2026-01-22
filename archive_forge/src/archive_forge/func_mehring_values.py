from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core import Site, Species
from pymatgen.core.tensors import SquareTensor
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
@property
def mehring_values(self):
    """Returns: the Chemical shielding tensor in Mehring Notation."""
    pas = self.principal_axis_system
    sigma_iso = pas.trace() / 3
    sigma_11, sigma_22, sigma_33 = np.diag(pas)
    return self.MehringNotation(sigma_iso, sigma_11, sigma_22, sigma_33)