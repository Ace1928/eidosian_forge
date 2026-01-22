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
def get_seebeck_eff_mass(self, output='average', temp=300, doping_levels=False, Lambda=0.5):
    """Seebeck effective mass calculated as explained in Ref.
        Gibbs, Z. M. et al., Effective mass and fermi surface complexity factor
        from ab initio band structure calculations.
        npj Computational Materials 3, 8 (2017).

        Args:
            output: 'average' returns the seebeck effective mass calculated using
                    the average of the three diagonal components of the seebeck tensor.
                    'tensor' returns the seebeck effective mass respect to the three
                    diagonal components of the seebeck tensor.
            doping_levels: False means that the seebeck effective mass is calculated
                           for every value of the chemical potential
                           True means that the seebeck effective mass is calculated
                           for every value of the doping levels for both n and p types
            temp:   temperature of calculated seebeck.
            Lambda: fitting parameter used to model the scattering (0.5 means constant
                    relaxation time).

        Returns:
            a list of values for the seebeck effective mass w.r.t the chemical potential,
            if doping_levels is set at False;
            a dict with n an p keys that contain a list of values for the seebeck effective
            mass w.r.t the doping levels, if doping_levels is set at True;
            if 'tensor' is selected, each element of the lists is a list containing
            the three components of the seebeck effective mass.
        """
    if doping_levels:
        sbk_mass = {}
        for dt in ('n', 'p'):
            concentrations = self.doping[dt]
            seebeck = self.get_seebeck(output=output, doping_levels=True)[dt][temp]
            sbk_mass[dt] = []
            for idx, concen in enumerate(concentrations):
                if output == 'average':
                    sbk_mass[dt].append(seebeck_eff_mass_from_seebeck_carr(abs(seebeck[idx]), concen, temp, Lambda))
                elif output == 'tensor':
                    sbk_mass[dt].append([])
                    for j in range(3):
                        sbk_mass[dt][-1].append(seebeck_eff_mass_from_seebeck_carr(abs(seebeck[idx][j][j]), concen, temp, Lambda))
    else:
        seebeck = self.get_seebeck(output=output, doping_levels=False)[temp]
        concentrations = self.get_carrier_concentration()[temp]
        sbk_mass = []
        for idx, concen in enumerate(concentrations):
            if output == 'average':
                sbk_mass.append(seebeck_eff_mass_from_seebeck_carr(abs(seebeck[idx]), concen, temp, Lambda))
            elif output == 'tensor':
                sbk_mass.append([])
                for j in range(3):
                    sbk_mass[-1].append(seebeck_eff_mass_from_seebeck_carr(abs(seebeck[idx][j][j]), concen, temp, Lambda))
    return sbk_mass