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
def read_relaxed(self, lines=None):
    """Check if ionic relaxation completed"""
    if not lines:
        lines = self.load_file('OUTCAR')
    for line in lines:
        if 'reached required accuracy' in line:
            return True
    return False