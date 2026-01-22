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
def seebeck_eff_mass_from_carr(eta, n, T, Lambda):
    """Calculate seebeck effective mass at a certain carrier concentration
    eta in kB*T units, n in cm-3, T in K, returns mass in m0 units.
    """
    try:
        from fdint import fdk
    except ImportError:
        raise BoltztrapError('fdint module not found. Please, install it.\nIt is needed to calculate Fermi integral quickly.')
    return (2 * np.pi ** 2 * abs(n) * 10 ** 6 / fdk(0.5, eta)) ** (2.0 / 3) / (2 * constants.m_e * constants.k * T / (constants.h / 2 / np.pi) ** 2)