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
def get_smeared_densities(self, sigma: float):
    """Returns the Dict representation of the densities, {Spin: densities},
        but with a Gaussian smearing of std dev sigma.

        Args:
            sigma: Std dev of Gaussian smearing function.

        Returns:
            Dict of Gaussian-smeared densities.
        """
    smeared_dens = {}
    diff = [self.energies[i + 1] - self.energies[i] for i in range(len(self.energies) - 1)]
    avg_diff = sum(diff) / len(diff)
    for spin, dens in self.densities.items():
        smeared_dens[spin] = gaussian_filter1d(dens, sigma / avg_diff)
    return smeared_dens