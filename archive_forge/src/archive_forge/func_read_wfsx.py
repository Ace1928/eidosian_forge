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
def read_wfsx(self):
    """
        Read the siesta WFSX file
        Return a namedtuple with the following arguments:
        """
    from ase.calculators.siesta.import_functions import readWFSX
    fname_woext = os.path.join(self.directory, self.prefix)
    if isfile(fname_woext + '.WFSX'):
        filename = fname_woext + '.WFSX'
        self.results['wfsx'] = readWFSX(filename)
    elif isfile(fname_woext + '.fullBZ.WFSX'):
        filename = fname_woext + '.fullBZ.WFSX'
        readWFSX(filename)
        self.results['wfsx'] = readWFSX(filename)
    else:
        self.results['wfsx'] = None