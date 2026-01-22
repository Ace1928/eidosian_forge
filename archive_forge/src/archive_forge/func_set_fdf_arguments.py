import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def set_fdf_arguments(self, fdf_arguments):
    """ Set the fdf_arguments after the initialization of the
            calculator.
        """
    self.validate_fdf_arguments(fdf_arguments)
    FileIOCalculator.set(self, fdf_arguments=fdf_arguments)