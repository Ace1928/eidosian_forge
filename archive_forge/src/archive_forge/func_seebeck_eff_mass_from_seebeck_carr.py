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
def seebeck_eff_mass_from_seebeck_carr(seeb, n, T, Lambda):
    """Find the chemical potential where analytic and calculated seebeck are identical
    and then calculate the seebeck effective mass at that chemical potential and
    a certain carrier concentration n.
    """
    eta = eta_from_seebeck(seeb, Lambda)
    return seebeck_eff_mass_from_carr(eta, n, T, Lambda)