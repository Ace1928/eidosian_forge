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
def get_site_spd_dos(self, site: PeriodicSite) -> dict[OrbitalType, Dos]:
    """Get orbital projected Dos of a particular site.

        Args:
            site: Site in Structure associated with CompleteDos.

        Returns:
            dict of {OrbitalType: Dos}, e.g. {OrbitalType.s: Dos object, ...}
        """
    spd_dos: dict[OrbitalType, dict[Spin, np.ndarray]] = {}
    for orb, pdos in self.pdos[site].items():
        orbital_type = _get_orb_type(orb)
        if orbital_type in spd_dos:
            spd_dos[orbital_type] = add_densities(spd_dos[orbital_type], pdos)
        else:
            spd_dos[orbital_type] = pdos
    return {orb: Dos(self.efermi, self.energies, densities) for orb, densities in spd_dos.items()}