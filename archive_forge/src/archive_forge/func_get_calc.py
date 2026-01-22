import pytest
from ase import Atoms
from ase.io import read
from ase.calculators.gaussian import Gaussian, GaussianOptimizer, GaussianIRC
from ase.optimize import LBFGS
def get_calc(**kwargs):
    kwargs.update(mem='100MW', method='hf', basis='sto-3g')
    return Gaussian(**kwargs)