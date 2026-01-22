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
def get_spd_dos(self) -> dict[str, Dos]:
    """Get orbital projected Dos.
        For example, if 3s and 4s are included in the basis of some element, they will be both summed in the orbital
        projected DOS.

        Returns:
            dict of {orbital: Dos}, e.g. {"s": Dos object, ...}
        """
    spd_dos = {}
    for atom_dos in self.pdos.values():
        for orb, pdos in atom_dos.items():
            orbital_type = _get_orb_type_lobster(orb)
            if orbital_type not in spd_dos:
                spd_dos[orbital_type] = pdos
            else:
                spd_dos[orbital_type] = add_densities(spd_dos[orbital_type], pdos)
    return {orb: Dos(self.efermi, self.energies, densities) for orb, densities in spd_dos.items()}