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
def get_site_orbital_dos(self, site: PeriodicSite, orbital: str) -> Dos:
    """Get the Dos for a particular orbital of a particular site.

        Args:
            site: Site in Structure associated with CompleteDos.
            orbital: principal quantum number and orbital in string format, e.g. "4s".
                    possible orbitals are: "s", "p_y", "p_z", "p_x", "d_xy", "d_yz", "d_z^2",
                    "d_xz", "d_x^2-y^2", "f_y(3x^2-y^2)", "f_xyz",
                    "f_yz^2", "f_z^3", "f_xz^2", "f_z(x^2-y^2)", "f_x(x^2-3y^2)"
                    In contrast to the Cohpcar and the Cohplist objects, the strings from the Lobster files are used

        Returns:
            Dos containing densities of an orbital of a specific site.
        """
    if orbital[1:] not in ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z^2', 'd_xz', 'd_x^2-y^2', 'f_y(3x^2-y^2)', 'f_xyz', 'f_yz^2', 'f_z^3', 'f_xz^2', 'f_z(x^2-y^2)', 'f_x(x^2-3y^2)']:
        raise ValueError('orbital is not correct')
    return Dos(self.efermi, self.energies, self.pdos[site][orbital])