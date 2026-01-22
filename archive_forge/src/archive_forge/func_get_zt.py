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
def get_zt(self, output='eigs', doping_levels=True, relaxation_time=1e-14, k_l=1):
    """Gives the ZT coefficient (S^2*cond*T/thermal cond) in either a full
        3x3 tensor form,
        as 3 eigenvalues, or as the average value (trace/3.0) If
        doping_levels=True, the results are given at
        different p and n doping levels (given by self.doping), otherwise it
        is given as a series of
        electron chemical potential values. We assume a constant relaxation
        time and a constant
        lattice thermal conductivity.

        Args:
            output (str): the type of output. 'tensor' give the full 3x3
            tensor, 'eigs' its 3 eigenvalues and
            'average' the average of the three eigenvalues
            doping_levels (bool): True for the results to be given at
            different doping levels, False for results
            at different electron chemical potentials
            relaxation_time (float): constant relaxation time in secs
            k_l (float): lattice thermal cond in W/(m*K)

        Returns:
            If doping_levels=True, a dictionary {temp:{'p':[],'n':[]}}. The
            'p' links to ZT
            at p-type doping and 'n' to the ZT at n-type doping. Otherwise,
            returns a {temp:[]} dictionary. The result contains either the
            sorted three eigenvalues of the symmetric
            ZT tensor (format='eigs') or a full tensor (3x3 array) (
            output='tensor') or as an average
            (output='average').
            The result includes a given constant relaxation time and lattice
            thermal conductivity
        """
    result = result_doping = None
    if doping_levels:
        result_doping = {doping: {t: [] for t in self._seebeck_doping[doping]} for doping in self._seebeck_doping}
        for doping in result_doping:
            for t in result_doping[doping]:
                for i in range(len(self.doping[doping])):
                    pf_tensor = np.dot(self._cond_doping[doping][t][i], np.dot(self._seebeck_doping[doping][t][i], self._seebeck_doping[doping][t][i]))
                    thermal_conduct = (self._kappa_doping[doping][t][i] - pf_tensor * t) * relaxation_time
                    result_doping[doping][t].append(np.dot(pf_tensor * relaxation_time * t, np.linalg.inv(thermal_conduct + k_l * np.eye(3, 3))))
    else:
        result = {t: [] for t in self._seebeck}
        for t in result:
            for i in range(len(self.mu_steps)):
                pf_tensor = np.dot(self._cond[t][i], np.dot(self._seebeck[t][i], self._seebeck[t][i]))
                thermal_conduct = (self._kappa[t][i] - pf_tensor * t) * relaxation_time
                result[t].append(np.dot(pf_tensor * relaxation_time * t, np.linalg.inv(thermal_conduct + k_l * np.eye(3, 3))))
    return BoltztrapAnalyzer._format_to_output(result, result_doping, output, doping_levels)