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
def virials(self) -> np.ndarray:
    """Returns virial tensor of each ionic step structure contained in MOVEMENT.

        Returns:
            np.ndarray: The virial tensor of each ionic step structure,
                with shape of (n_ionic_steps, 3, 3)
        """
    return np.array([step['virial'] for step in self.ionic_steps if 'virial' in step])