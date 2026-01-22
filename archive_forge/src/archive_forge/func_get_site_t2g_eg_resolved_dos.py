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
def get_site_t2g_eg_resolved_dos(self, site: PeriodicSite) -> dict[str, Dos]:
    """Get the t2g, eg projected DOS for a particular site.

        Args:
            site: Site in Structure associated with CompleteDos.

        Returns:
            A dict {"e_g": Dos, "t2g": Dos} containing summed e_g and t2g DOS
            for the site.
        """
    warnings.warn('Are the orbitals correctly oriented? Are you sure?')
    t2g_dos = []
    eg_dos = []
    for s, atom_dos in self.pdos.items():
        if s == site:
            for orb, pdos in atom_dos.items():
                if _get_orb_lobster(orb) in (Orbital.dxy, Orbital.dxz, Orbital.dyz):
                    t2g_dos.append(pdos)
                elif _get_orb_lobster(orb) in (Orbital.dx2, Orbital.dz2):
                    eg_dos.append(pdos)
    return {'t2g': Dos(self.efermi, self.energies, functools.reduce(add_densities, t2g_dos)), 'e_g': Dos(self.efermi, self.energies, functools.reduce(add_densities, eg_dos))}