from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
@staticmethod
def fp_to_dict(fp: NamedTuple) -> dict:
    """Converts a fingerprint into a dictionary.

        Args:
            fp: The DOS fingerprint to be converted into a dictionary

        Returns:
            dict: A dict of the fingerprint Keys=type, Values=np.ndarray(energies, densities)
        """
    fp_dict = {}
    fp_dict[fp[2]] = np.array([fp[0], fp[1]], dtype='object').T
    return fp_dict