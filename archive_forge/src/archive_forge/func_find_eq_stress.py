from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
def find_eq_stress(strains, stresses, tol: float=1e-10):
    """
    Finds stress corresponding to zero strain state in stress-strain list.

    Args:
        strains (Nx3x3 array-like): array corresponding to strains
        stresses (Nx3x3 array-like): array corresponding to stresses
        tol (float): tolerance to find zero strain state
    """
    stress_array = np.array(stresses)
    strain_array = np.array(strains)
    eq_stress = stress_array[np.all(abs(strain_array) < tol, axis=(1, 2))]
    if eq_stress.size != 0:
        all_same = (abs(eq_stress - eq_stress[0]) < 1e-08).all()
        if len(eq_stress) > 1 and (not all_same):
            raise ValueError('Multiple stresses found for equilibrium strain state, please specify equilibrium stress or   remove extraneous stresses.')
        eq_stress = eq_stress[0]
    else:
        warnings.warn('No eq state found, returning zero voigt stress')
        eq_stress = Stress(np.zeros((3, 3)))
    return eq_stress