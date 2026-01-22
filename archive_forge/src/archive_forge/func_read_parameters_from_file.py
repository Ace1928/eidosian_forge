import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError, Parameters
from ase.units import kcal, mol, Debye
def read_parameters_from_file(self, lines):
    """Find and return the line that defines a Mopac calculation

        Parameters:

            lines: list of str
        """
    for i, line in enumerate(lines):
        if line.find('CALCULATION DONE:') != -1:
            break
    lines1 = lines[i:]
    for i, line in enumerate(lines1):
        if line.find('****') != -1:
            return lines1[i + 1]