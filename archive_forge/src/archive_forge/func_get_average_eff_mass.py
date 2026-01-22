from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def get_average_eff_mass(self, output='eigs', doping_levels=True):
    """Gives the average effective mass tensor. We call it average because
        it takes into account all the bands
        and regions in the Brillouin zone. This is different than the standard
        textbook effective mass which relates
        often to only one (parabolic) band.
        The average effective mass tensor is defined as the integrated
        average of the second derivative of E(k)
        This effective mass tensor takes into account:
        -non-parabolicity
        -multiple extrema
        -multiple bands.

        For more information about it. See:

        Hautier, G., Miglio, A., Waroquiers, D., Rignanese, G., & Gonze,
        X. (2014).
        How Does Chemistry Influence Electron Effective Mass in Oxides?
        A High-Throughput Computational Analysis. Chemistry of Materials,
        26(19), 5447-5458. doi:10.1021/cm404079a

        or

        Hautier, G., Miglio, A., Ceder, G., Rignanese, G.-M., & Gonze,
        X. (2013).
        Identification and design principles of low hole effective mass
        p-type transparent conducting oxides.
        Nature Communications, 4, 2292. doi:10.1038/ncomms3292

        Depending on the value of output, we have either the full 3x3
        effective mass tensor,
        its 3 eigenvalues or an average

        Args:
            output (str): 'eigs' for eigenvalues, 'tensor' for the full
            tensor and 'average' for an average (trace/3)
            doping_levels (bool): True for the results to be given at
            different doping levels, False for results
            at different electron chemical potentials

        Returns:
            If doping_levels=True,a dictionary {'p':{temp:[]},'n':{temp:[]}}
            with an array of effective mass tensor, eigenvalues of average
            value (depending on output) for each temperature and for each
            doping level.
            The 'p' links to hole effective mass tensor and 'n' to electron
            effective mass tensor.
        """
    result = result_doping = None
    conc = self.get_carrier_concentration()
    if doping_levels:
        result_doping = {doping: {t: [] for t in self._cond_doping[doping]} for doping in self.doping}
        for doping in result_doping:
            for temp in result_doping[doping]:
                for i in range(len(self.doping[doping])):
                    try:
                        result_doping[doping][temp].append(np.linalg.inv(np.array(self._cond_doping[doping][temp][i])) * self.doping[doping][i] * 10 ** 6 * constants.e ** 2 / constants.m_e)
                    except np.linalg.LinAlgError:
                        pass
    else:
        result = {t: [] for t in self._seebeck}
        for temp in result:
            for i in range(len(self.mu_steps)):
                try:
                    cond_inv = np.linalg.inv(np.array(self._cond[temp][i]))
                except np.linalg.LinAlgError:
                    pass
                result[temp].append(cond_inv * conc[temp][i] * 10 ** 6 * constants.e ** 2 / constants.m_e)
    return BoltztrapAnalyzer._format_to_output(result, result_doping, output, doping_levels)