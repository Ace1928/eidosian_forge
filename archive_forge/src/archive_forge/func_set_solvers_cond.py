from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def set_solvers_cond(self, solvers):
    """ Sets the solver strings used in this dat file """
    self.solvers_cond = deepcopy(solvers)