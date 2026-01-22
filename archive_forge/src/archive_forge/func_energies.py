import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.calculators.calculator import PropertyNotImplementedError
@property
def energies(self) -> np.ndarray:
    """The energies of this band structure.

        This is a numpy array of shape (nspins, nkpoints, nbands)."""
    return self._energies