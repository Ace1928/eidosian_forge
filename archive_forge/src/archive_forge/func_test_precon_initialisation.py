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
def test_precon_initialisation(setup_images):
    images, _, _ = setup_images
    mep = NEB(images, method='spline', precon='Exp')
    mep.get_forces()
    assert len(mep.precon) == len(mep.images)
    assert mep.precon[0].mu == mep.precon[1].mu