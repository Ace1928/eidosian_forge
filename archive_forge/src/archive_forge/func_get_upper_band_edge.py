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
def get_upper_band_edge(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None) -> float:
    """Get the orbital-projected upper band edge. The definition by Xin et al.
        Phys. Rev. B, 89, 115114 (2014) is used, which is the highest peak position of the
        Hilbert transform of the orbital-projected DOS.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.

        Returns:
            Upper band edge in eV, often denoted epsilon_u
        """
    transformed_dos = self.get_hilbert_transform(elements=elements, sites=sites, band=band)
    energies = transformed_dos.energies - transformed_dos.efermi
    densities = transformed_dos.get_densities(spin=spin)
    if erange:
        densities = densities[(energies >= erange[0]) & (energies <= erange[1])]
        energies = energies[(energies >= erange[0]) & (energies <= erange[1])]
    return energies[np.argmax(densities)]