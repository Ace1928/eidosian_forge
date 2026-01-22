import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def read_forces(self, all=False, lines=None):
    """Method that reads forces from OUTCAR file.

        If 'all' is switched on, the forces for all ionic steps
        in the OUTCAR file be returned, in other case only the
        forces for the last ionic configuration is returned."""
    if not lines:
        lines = self.load_file('OUTCAR')
    if all:
        all_forces = []
    for n, line in enumerate(lines):
        if 'TOTAL-FORCE' in line:
            forces = []
            for i in range(len(self.atoms)):
                forces.append(np.array([float(f) for f in lines[n + 2 + i].split()[3:6]]))
            if all:
                all_forces.append(np.array(forces)[self.resort])
    if all:
        return np.array(all_forces)
    return np.array(forces)[self.resort]