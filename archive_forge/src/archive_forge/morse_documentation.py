import numpy as np
from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list

        Parameters
        ----------
        epsilon: float
          Absolute minimum depth, default 1.0
        r0: float
          Minimum distance, default 1.0
        rho0: float
          Exponential prefactor. The force constant in the potential minimum
          is k = 2 * epsilon * (rho0 / r0)**2, default 6.0
        