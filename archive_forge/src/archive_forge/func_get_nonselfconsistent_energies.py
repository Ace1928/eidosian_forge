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
def get_nonselfconsistent_energies(self, bee_type):
    """ Method that reads and returns BEE energy contributions
            written in OUTCAR file.
        """
    assert bee_type == 'beefvdw'
    cmd = 'grep -32 "BEEF xc energy contributions" OUTCAR | tail -32'
    p = os.popen(cmd, 'r')
    s = p.readlines()
    p.close()
    xc = np.array([])
    for line in s:
        l_ = float(line.split(':')[-1])
        xc = np.append(xc, l_)
    assert len(xc) == 32
    return xc