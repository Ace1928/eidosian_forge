from __future__ import annotations
import linecache
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.kpath import KPathSeek
def get_n_atoms(self) -> int:
    """Return the number of atoms in structure.

        Returns:
            int: The number of atoms
        """
    return int(self.strs_lst[0].split()[0].strip())