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
def read_stress(self, lines=None):
    """Read stress from OUTCAR.

        Depreciated: Use get_stress() instead.
        """
    if not lines:
        lines = self.load_file('OUTCAR')
    stress = None
    for line in lines:
        if ' in kB  ' in line:
            stress = -np.array([float(a) for a in line.split()[2:]])
            stress = stress[[0, 1, 2, 4, 5, 3]] * 0.1 * ase.units.GPa
    return stress