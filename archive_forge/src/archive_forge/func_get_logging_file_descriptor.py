import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from scipy.special import erfinv, erfc
from ase.neighborlist import neighbor_list
from ase.parallel import world
from ase.utils import IOContext
def get_logging_file_descriptor(calculator):
    if hasattr(calculator, 'log'):
        fd = calculator.log
        if hasattr(fd, 'write'):
            return fd
        if hasattr(fd, 'fd'):
            return fd.fd
    if hasattr(calculator, 'txt'):
        return calculator.txt