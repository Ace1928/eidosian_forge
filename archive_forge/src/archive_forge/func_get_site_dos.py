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
def get_site_dos(self, site: PeriodicSite) -> Dos:
    """Get the total Dos for a site (all orbitals).

        Args:
            site: Site in Structure associated with CompleteDos.

        Returns:
            Dos containing summed orbital densities for site.
        """
    site_dos = functools.reduce(add_densities, self.pdos[site].values())
    return Dos(self.efermi, self.energies, site_dos)