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
def e_tots(self) -> np.ndarray:
    """Returns total energies of each ionic step structures contained in MOVEMENT.

        Returns:
            np.ndarray: Total energy of of each ionic step structure,
                with shape of (n_ionic_steps,).
        """
    return np.array([step['e_tot'] for step in self.ionic_steps])