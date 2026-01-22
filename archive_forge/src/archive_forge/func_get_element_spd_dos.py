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
def get_element_spd_dos(self, el: SpeciesLike) -> dict[str, Dos]:
    """Get element and spd projected Dos.

        Args:
            el: Element in Structure.composition associated with LobsterCompleteDos

        Returns:
            dict of {OrbitalType.s: densities, OrbitalType.p: densities, OrbitalType.d: densities}
        """
    el = get_el_sp(el)
    el_dos = {}
    for site, atom_dos in self.pdos.items():
        if site.specie == el:
            for orb, pdos in atom_dos.items():
                orbital_type = _get_orb_type_lobster(orb)
                if orbital_type not in el_dos:
                    el_dos[orbital_type] = pdos
                else:
                    el_dos[orbital_type] = add_densities(el_dos[orbital_type], pdos)
    return {orb: Dos(self.efermi, self.energies, densities) for orb, densities in el_dos.items()}