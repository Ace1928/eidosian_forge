from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
def cv(self, temp: float | None=None, structure: Structure | None=None, **kwargs) -> float:
    """Constant volume specific heat C_v at temperature T obtained from the integration of the DOS.
        Only positive frequencies will be used.
        Result in J/(K*mol-c). A mol-c is the abbreviation of a mole-cell, that is, the number
        of Avogadro times the atoms in a unit cell. To compare with experimental data the result
        should be divided by the number of unit formulas in the cell. If the structure is provided
        the division is performed internally and the result is in J/(K*mol).

        Args:
            temp: a temperature in K
            structure: the structure of the system. If not None it will be used to determine the number of
                formula units
            **kwargs: allows passing in deprecated t parameter for temp

        Returns:
            float: Constant volume specific heat C_v
        """
    temp = kwargs.get('t', temp)
    if temp == 0:
        return 0
    freqs = self._positive_frequencies
    dens = self._positive_densities

    def csch2(x):
        return 1.0 / np.sinh(x) ** 2
    wd2kt = freqs / (2 * BOLTZ_THZ_PER_K * temp)
    cv = np.trapz(wd2kt ** 2 * csch2(wd2kt) * dens, x=freqs)
    cv *= const.Boltzmann * const.Avogadro
    if structure:
        formula_units = structure.composition.num_atoms / structure.composition.reduced_composition.num_atoms
        cv /= formula_units
    return cv