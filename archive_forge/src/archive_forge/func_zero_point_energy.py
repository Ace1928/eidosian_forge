from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
def zero_point_energy(self, structure: Structure | None=None) -> float:
    """Zero point energy of the system. Only positive frequencies will be used.
        Result in J/mol-c. A mol-c is the abbreviation of a mole-cell, that is, the number
        of Avogadro times the atoms in a unit cell. To compare with experimental data the result
        should be divided by the number of unit formulas in the cell. If the structure is provided
        the division is performed internally and the result is in J/mol.

        Args:
            structure: the structure of the system. If not None it will be used to determine the number of
                formula units

        Returns:
            Phonon contribution to the internal energy
        """
    freqs = self._positive_frequencies
    dens = self._positive_densities
    zpe = 0.5 * np.trapz(freqs * dens, x=freqs)
    zpe *= THZ_TO_J * const.Avogadro
    if structure:
        formula_units = structure.composition.num_atoms / structure.composition.reduced_composition.num_atoms
        zpe /= formula_units
    return zpe