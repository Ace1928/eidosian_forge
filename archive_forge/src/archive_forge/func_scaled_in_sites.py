from typing import List
import numpy as np
from ase import Atoms
from .spacegroup import Spacegroup, _SPACEGROUP
def scaled_in_sites(scaled_pos: np.ndarray, sites: np.ndarray):
    """Check if a scaled position is in a site"""
    for site in sites:
        if np.allclose(site, scaled_pos, atol=tol):
            return True
    return False