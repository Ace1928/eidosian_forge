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
def test_integrate_forces(setup_images):
    images, _, _ = setup_images
    forcefit = fit_images(images)
    neb = NEB(images)
    spline_points = 1000
    s, E, F = neb.integrate_forces(spline_points=spline_points)
    np.testing.assert_allclose(E[0] - E[-1], forcefit.energies[0] - forcefit.energies[-1], atol=1e-10)
    assert np.argmax(E) == spline_points // 2 - 1
    np.testing.assert_allclose(E.max(), forcefit.energies.max(), rtol=0.025)