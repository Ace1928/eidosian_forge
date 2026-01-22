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
def atom_forces(self) -> np.ndarray:
    """Returns forces on atoms in each structures contained in MOVEMENT.

        Returns:
            np.ndarray: The forces on atoms of each ionic step structure,
                with shape of (n_ionic_steps, n_atoms, 3).
        """
    return np.array([step['atom_forces'] for step in self.ionic_steps])