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
def get_complexity_factor(self, output='average', temp=300, doping_levels=False, Lambda=0.5):
    """Fermi surface complexity factor respect to calculated as explained in Ref.
        Gibbs, Z. M. et al., Effective mass and fermi surface complexity factor
        from ab initio band structure calculations.
        npj Computational Materials 3, 8 (2017).

        Args:
            output: 'average' returns the complexity factor calculated using the average
                    of the three diagonal components of the seebeck and conductivity tensors.
                    'tensor' returns the complexity factor respect to the three
                    diagonal components of seebeck and conductivity tensors.
            doping_levels: False means that the complexity factor is calculated
                           for every value of the chemical potential
                           True means that the complexity factor is calculated
                           for every value of the doping levels for both n and p types
            temp:   temperature of calculated seebeck and conductivity.
            Lambda: fitting parameter used to model the scattering (0.5 means constant
                    relaxation time).

        Returns:
            a list of values for the complexity factor w.r.t the chemical potential,
            if doping_levels is set at False;
            a dict with n an p keys that contain a list of values for the complexity factor
            w.r.t the doping levels, if doping_levels is set at True;
            if 'tensor' is selected, each element of the lists is a list containing
            the three components of the complexity factor.
        """
    if doping_levels:
        cmplx_fact = {}
        for dt in ('n', 'p'):
            sbk_mass = self.get_seebeck_eff_mass(output, temp, doping_levels=True, Lambda=Lambda)[dt]
            cond_mass = self.get_average_eff_mass(output=output, doping_levels=True)[dt][temp]
            if output == 'average':
                cmplx_fact[dt] = [(m_s / abs(m_c)) ** 1.5 for m_s, m_c in zip(sbk_mass, cond_mass)]
            elif output == 'tensor':
                cmplx_fact[dt] = []
                for i, sm in enumerate(sbk_mass):
                    cmplx_fact[dt].append([])
                    for j in range(3):
                        cmplx_fact[dt][-1].append((sm[j] / abs(cond_mass[i][j][j])) ** 1.5)
    else:
        sbk_mass = self.get_seebeck_eff_mass(output, temp, doping_levels=False, Lambda=Lambda)
        cond_mass = self.get_average_eff_mass(output=output, doping_levels=False)[temp]
        if output == 'average':
            cmplx_fact = [(m_s / abs(m_c)) ** 1.5 for m_s, m_c in zip(sbk_mass, cond_mass)]
        elif output == 'tensor':
            cmplx_fact = []
            for i, sm in enumerate(sbk_mass):
                cmplx_fact.append([])
                for j in range(3):
                    cmplx_fact[-1].append((sm[j] / abs(cond_mass[i][j][j])) ** 1.5)
    return cmplx_fact