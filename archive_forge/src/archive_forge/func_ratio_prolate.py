from __future__ import annotations
import re
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter, str2float
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.util.due import Doi, due
@property
def ratio_prolate(self) -> np.ndarray:
    """This will compute ratio between largest and smallest eigenvalue of Ucart."""
    ratios = []
    for us in self.U1U2U3:
        ratios.append(np.max(us) / np.min(us))
    return np.array(ratios)