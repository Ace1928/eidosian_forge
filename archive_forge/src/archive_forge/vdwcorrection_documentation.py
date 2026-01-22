import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from scipy.special import erfinv, erfc
from ase.neighborlist import neighbor_list
from ase.parallel import world
from ase.utils import IOContext
Damping factor.

        Standard values for d and sR as given in
        Tkatchenko and Scheffler PRL 102 (2009) 073005.