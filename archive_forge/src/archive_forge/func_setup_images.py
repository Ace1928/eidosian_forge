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
def setup_images(_setup_images_global):
    images, i1, i2 = _setup_images_global
    new_images = [img.copy() for img in images]
    for img in new_images:
        img.calc = calc()
    return (new_images, i1, i2)