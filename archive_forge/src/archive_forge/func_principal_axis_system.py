from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core import Site, Species
from pymatgen.core.tensors import SquareTensor
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
@property
def principal_axis_system(self):
    """
        Returns a electric field gradient tensor aligned to the principle axis system so that only the 3 diagonal
        components are non-zero.
        """
    return ElectricFieldGradient(np.diag(np.sort(np.linalg.eigvals(self))))