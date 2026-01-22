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
def compare_dict(d1, d2):
    """Helper function to compare dictionaries"""
    if set(d1.keys()) ^ set(d2.keys()):
        return False
    for key, value in d1.items():
        if np.any(value != d2[key]):
            return False
    return True