from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def set_pseudos(self, pots):
    """ Sets the pseudopotential files used in this dat file """
    self.pseudos = deepcopy(pots)