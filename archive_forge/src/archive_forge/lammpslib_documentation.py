import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
Determine LAMMPS boundary types based on ASE pbc settings. For
        non-periodic dimensions, if the cell length is finite then
        fixed BCs ('f') are used; if the cell length is approximately
        zero, shrink-wrapped BCs ('s') are used.