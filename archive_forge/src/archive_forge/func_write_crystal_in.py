from ase.units import Hartree, Bohr
from ase.io import write
import numpy as np
import os
from ase.calculators.calculator import FileIOCalculator
def write_crystal_in(self, filename):
    """ Write the input file for the crystal calculation.
            Geometry is taken always from the file 'fort.34'
        """
    with open(filename, 'wt', encoding='latin-1') as outfile:
        self._write_crystal_in(outfile)