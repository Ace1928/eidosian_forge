import os
from typing import Any, Union
import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import world, parprint
def ensemble(energy: float, contributions: np.ndarray, xc: str, verbose: bool=False) -> np.ndarray:
    """Returns an array of ensemble total energies."""
    ensemble = BEEFEnsemble(None, energy, contributions, xc, verbose)
    return ensemble.get_ensemble_energies()