import json
import numpy as np
import pytest
from ase.calculators.morse import MorsePotential
from ase.optimize import BFGS, ODE12r
from ase.optimize.precon import Exp
from ase.build import bulk
from ase.neb import NEB, NEBTools, NEBOptimizer
from ase.geometry.geometry import find_mic
from ase.constraints import FixBondLength
from ase.geometry.geometry import get_distances
from ase.utils.forcecurve import fit_images
@pytest.fixture
def ref_vacancy(_ref_vacancy_global):
    Ef_ref, dE_ref, saddle = _ref_vacancy_global
    return (Ef_ref, dE_ref, saddle.copy())