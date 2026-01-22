import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def print_input(self, debug=None, nohup=None):
    """
        For a debugging purpose, print the .dat file
        """
    if debug is None:
        debug = self.debug
    if nohup is None:
        nohup = self.nohup
    self.prind('Reading input file' + self.label)
    filename = get_file_name('.dat', self.label)
    if not nohup:
        with open(filename, 'r') as fd:
            while True:
                line = fd.readline()
                print(line.strip())
                if not line:
                    break