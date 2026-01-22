import re
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import Prism, convert
from ase.utils import reader, writer
Write atomic structure data to a LAMMPS data file.