from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
@property
def e_atoms(self) -> np.ndarray:
    """
        Returns individual energies of atoms in each ionic step structures
        contained in MOVEMENT.

        Returns:
            np.ndarray: The individual energy of atoms in each ionic step structure,
                with shape of (n_ionic_steps, n_atoms).
        """
    return np.array([step['eatoms'] for step in self.ionic_steps if 'eatoms' in step])