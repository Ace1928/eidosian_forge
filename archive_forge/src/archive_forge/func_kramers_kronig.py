from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants
import scipy.special
from monty.json import MSONable
from tqdm import tqdm
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder
def kramers_kronig(eps: np.ndarray, nedos: int, deltae: float, cshift: float=0.1) -> NDArray:
    """Perform the Kramers-Kronig transformation.

    Perform the Kramers-Kronig transformation exactly as VASP does it.
    The input eps should be complex and the imaginary part of the dielectric function
    should be stored as the real part of the complex input array.
    The output should be the complex dielectric function.

    Args:
        eps: The dielectric function with the imaginary part stored as the real part and nothing in the imaginary part.
        nedos: The sampling of the energy values
        deltae: The energy grid spacing
        cshift: The shift of the imaginary part of the dielectric function.

    Returns:
        np.array: Array of size `nedos` with the complex dielectric function.
    """
    egrid = np.linspace(0, deltae * nedos, nedos)
    csfhit = cshift * 1j
    cdiff = np.subtract.outer(egrid, egrid) + csfhit
    csum = np.add.outer(egrid, egrid) + csfhit
    vals = -0.5 * (eps / cdiff - np.conj(eps) / csum)
    return np.sum(vals, axis=1) * 2 / np.pi * deltae