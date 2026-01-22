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
def read_spinpol(self, lines=None):
    """Method which reads if a calculation from spinpolarized using OUTCAR.

        Depreciated: Use get_spin_polarized() instead.
        """
    if not lines:
        lines = self.load_file('OUTCAR')
    for line in lines:
        if 'ISPIN' in line:
            if int(line.split()[2]) == 2:
                self.spinpol = True
            else:
                self.spinpol = False
    return self.spinpol