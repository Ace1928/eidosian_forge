from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.dev import requires
from monty.json import MSONable
from scipy.interpolate import UnivariateSpline
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import amu_to_kg
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
def thermal_conductivity_slack(self, squared: bool=True, limit_frequencies: Literal['debye', 'acoustic'] | None=None, theta_d: float | None=None, t: float | None=None) -> float:
    """Calculates the thermal conductivity at the acoustic Debye temperature with the Slack formula,
        using the average Gruneisen.
        Adapted from abipy.

        Args:
            squared (bool): if True the average is performed on the squared values of the Gruenisen
            limit_frequencies: if None (default) no limit on the frequencies will be applied.
                Possible values are "debye" (only modes with frequencies lower than the acoustic Debye
                temperature) and "acoustic" (only the acoustic modes, i.e. the first three modes).
            theta_d: the temperature used to estimate the average of the Gruneisen used in the
                Slack formula. If None the acoustic Debye temperature is used (see
                acoustic_debye_temp). Will also be considered as the Debye temperature in the
                Slack formula.
            t: temperature at which the thermal conductivity is estimated. If None the value at
                the calculated acoustic Debye temperature is given. The value is obtained as a
                simple rescaling of the value at the Debye temperature.

        Returns:
            The value of the thermal conductivity in W/(m*K)
        """
    assert self.structure is not None, 'Structure is not defined.'
    average_mass = np.mean([s.specie.atomic_mass for s in self.structure]) * amu_to_kg
    if theta_d is None:
        theta_d = self.acoustic_debye_temp
    mean_g = self.average_gruneisen(t=theta_d, squared=squared, limit_frequencies=limit_frequencies)
    f1 = 0.849 * 3 * 4 ** (1 / 3) / (20 * np.pi ** 3 * (1 - 0.514 * mean_g ** (-1) + 0.228 * mean_g ** (-2)))
    f2 = (const.k * theta_d / const.hbar) ** 2
    f3 = const.k * average_mass * self.structure.volume ** (1 / 3) * 1e-10 / (const.hbar * mean_g ** 2)
    k = f1 * f2 * f3
    if t is not None:
        k *= theta_d / t
    return k