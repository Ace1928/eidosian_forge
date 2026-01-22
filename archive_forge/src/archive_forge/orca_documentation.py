import os
import re
import numpy as np
from ase.units import Hartree, Bohr
from ase.io.orca import write_orca
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
 reads forces on point charges from .pcgrad file 