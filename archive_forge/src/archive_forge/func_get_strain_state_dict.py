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
def get_strain_state_dict(strains, stresses, eq_stress=None, tol: float=1e-10, add_eq=True, sort=True):
    """
    Creates a dictionary of voigt notation stress-strain sets
    keyed by "strain state", i. e. a tuple corresponding to
    the non-zero entries in ratios to the lowest nonzero value,
    e.g. [0, 0.1, 0, 0.2, 0, 0] -> (0,1,0,2,0,0)
    This allows strains to be collected in stencils as to
    evaluate parameterized finite difference derivatives.

    Args:
        strains (Nx3x3 array-like): strain matrices
        stresses (Nx3x3 array-like): stress matrices
        eq_stress (Nx3x3 array-like): equilibrium stress
        tol (float): tolerance for sorting strain states
        add_eq (bool): Whether to add eq_strain to stress-strain sets for each strain state.
            Defaults to True.
        sort (bool): flag for whether to sort strain states

    Returns:
        dict: strain state keys and dictionaries with stress-strain data corresponding to strain state
    """
    vstrains = np.array([Strain(s).zeroed(tol).voigt for s in strains])
    vstresses = np.array([Stress(s).zeroed(tol).voigt for s in stresses])
    independent = {tuple(np.nonzero(vstrain)[0].tolist()) for vstrain in vstrains}
    strain_state_dict = {}
    if add_eq:
        veq_stress = Stress(eq_stress).voigt if eq_stress is not None else find_eq_stress(strains, stresses).voigt
    for ind in independent:
        template = np.zeros(6, dtype=bool)
        np.put(template, ind, v=True)
        template = np.tile(template, [vstresses.shape[0], 1])
        mode = (template == (np.abs(vstrains) > 1e-10)).all(axis=1)
        mstresses = vstresses[mode]
        mstrains = vstrains[mode]
        min_nonzero_ind = np.argmin(np.abs(np.take(mstrains[-1], ind)))
        min_nonzero_val = np.take(mstrains[-1], ind)[min_nonzero_ind]
        strain_state = mstrains[-1] / min_nonzero_val
        strain_state = tuple(strain_state)
        if add_eq:
            mstrains = np.vstack([mstrains, np.zeros(6)])
            mstresses = np.vstack([mstresses, veq_stress])
        if sort:
            mstresses = mstresses[mstrains[:, ind[0]].argsort()]
            mstrains = mstrains[mstrains[:, ind[0]].argsort()]
        strain_state_dict[strain_state] = {'strains': mstrains, 'stresses': mstresses}
    return strain_state_dict