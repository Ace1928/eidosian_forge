from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def get_reasonable_repetitions(n_atoms: int) -> tuple[int, int, int]:
    """Choose the number of repetitions in a supercell
    according to the number of atoms in the system.
    """
    if n_atoms < 4:
        return (3, 3, 3)
    if 4 <= n_atoms < 15:
        return (2, 2, 2)
    if 15 <= n_atoms < 50:
        return (2, 2, 1)
    return (1, 1, 1)