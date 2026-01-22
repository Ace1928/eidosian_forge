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
def read_vib_freq(self, lines=None):
    """Read vibrational frequencies.

        Returns list of real and list of imaginary frequencies."""
    freq = []
    i_freq = []
    if not lines:
        lines = self.load_file('OUTCAR')
    for line in lines:
        data = line.split()
        if 'THz' in data:
            if 'f/i=' not in data:
                freq.append(float(data[-2]))
            else:
                i_freq.append(float(data[-2]))
    return (freq, i_freq)