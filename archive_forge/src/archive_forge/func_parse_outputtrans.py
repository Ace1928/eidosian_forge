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
@staticmethod
def parse_outputtrans(path_dir):
    """Parses .outputtrans file.

        Args:
            path_dir: dir containing boltztrap.outputtrans

        Returns:
            tuple - (run_type, warning, efermi, gap, doping_levels)
        """
    run_type = warning = efermi = gap = None
    doping_levels = []
    with open(f'{path_dir}/boltztrap.outputtrans') as file:
        for line in file:
            if 'WARNING' in line:
                warning = line
            elif 'Calc type:' in line:
                run_type = line.split()[-1]
            elif line.startswith('VBM'):
                efermi = Energy(line.split()[1], 'Ry').to('eV')
            elif line.startswith('Egap:'):
                gap = Energy(float(line.split()[1]), 'Ry').to('eV')
            elif line.startswith('Doping level number'):
                doping_levels.append(float(line.split()[6]))
    return (run_type, warning, efermi, gap, doping_levels)