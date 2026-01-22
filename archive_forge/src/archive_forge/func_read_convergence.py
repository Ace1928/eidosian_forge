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
def read_convergence(self, lines=None):
    """Method that checks whether a calculation has converged."""
    if not lines:
        lines = self.load_file('OUTCAR')
    converged = None
    for line in lines:
        if 0:
            if line.rfind('aborting loop') > -1:
                raise RuntimeError(line.strip())
                break
        if 'EDIFF  ' in line:
            ediff = float(line.split()[2])
        if 'total energy-change' in line:
            if 'MIXING' in line:
                continue
            split = line.split(':')
            a = float(split[1].split('(')[0])
            b = split[1].split('(')[1][0:-2]
            if 'e' not in b.lower():
                bsplit = b.split('-')
                bsplit[-1] = 'e' + bsplit[-1]
                b = '-'.join(bsplit).replace('-e', 'e-')
            b = float(b)
            if [abs(a), abs(b)] < [ediff, ediff]:
                converged = True
            else:
                converged = False
                continue
    if self.int_params['ibrion'] in [1, 2, 3] and self.int_params['nsw'] not in [0]:
        if not self.read_relaxed():
            converged = False
        else:
            converged = True
    return converged