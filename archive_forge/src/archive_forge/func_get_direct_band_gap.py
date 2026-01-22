from __future__ import annotations
import collections
import itertools
import math
import re
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff
def get_direct_band_gap(self):
    """Returns the direct band gap.

        Returns:
            the value of the direct band gap
        """
    if self.is_metal():
        return 0.0
    dg = self.get_direct_band_gap_dict()
    return min((v['value'] for v in dg.values()))