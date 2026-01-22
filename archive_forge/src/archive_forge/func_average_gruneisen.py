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
def average_gruneisen(self, t: float | None=None, squared: bool=True, limit_frequencies: Literal['debye', 'acoustic'] | None=None) -> float:
    """Calculates the average of the Gruneisen based on the values on the regular grid.
        If squared is True the average will use the squared value of the Gruneisen and a squared root
        is performed on the final result.
        Values associated to negative frequencies will be ignored.
        See Scripta Materialia 129, 88 for definitions.
        Adapted from classes in abipy that have been written by Guido Petretto (UCLouvain).

        Args:
            t: the temperature at which the average Gruneisen will be evaluated. If None the acoustic Debye
                temperature is used (see acoustic_debye_temp).
            squared: if True the average is performed on the squared values of the Grueneisen.
            limit_frequencies: if None (default) no limit on the frequencies will be applied.
                Possible values are "debye" (only modes with frequencies lower than the acoustic Debye
                temperature) and "acoustic" (only the acoustic modes, i.e. the first three modes).

        Returns:
            The average Gruneisen parameter
        """
    if t is None:
        t = self.acoustic_debye_temp
    w = self.frequencies
    wdkt = w * const.tera / (const.value('Boltzmann constant in Hz/K') * t)
    exp_wdkt = np.exp(wdkt)
    cv = np.choose(w > 0, (0, const.value('Boltzmann constant in eV/K') * wdkt ** 2 * exp_wdkt / (exp_wdkt - 1) ** 2))
    gamma = self.gruneisen
    if squared:
        gamma = gamma ** 2
    if limit_frequencies == 'debye':
        acoustic_debye_freq = self.acoustic_debye_temp * const.value('Boltzmann constant in Hz/K') / const.tera
        ind = np.where((w >= 0) & (w <= acoustic_debye_freq))
    elif limit_frequencies == 'acoustic':
        w_acoustic = w[:, :3]
        ind = np.where(w_acoustic >= 0)
    elif limit_frequencies is None:
        ind = np.where(w >= 0)
    else:
        raise ValueError(f'{limit_frequencies} is not an accepted value for limit_frequencies.')
    weights = self.multiplicities
    assert weights is not None, 'Multiplicities are not defined.'
    g = np.dot(weights[ind[0]], np.multiply(cv, gamma)[ind]).sum() / np.dot(weights[ind[0]], cv[ind]).sum()
    if squared:
        g = np.sqrt(g)
    return g