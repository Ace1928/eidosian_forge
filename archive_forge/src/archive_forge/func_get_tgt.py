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
def get_tgt(self, temperature: float | None=None, structure: Structure=None, quad=None):
    """
        Gets the thermodynamic Gruneisen tensor (TGT) by via an
        integration of the GGT weighted by the directional heat
        capacity.

        See refs:
            R. N. Thurston and K. Brugger, Phys. Rev. 113, A1604 (1964).
            K. Brugger Phys. Rev. 137, A1826 (1965).

        Args:
            temperature (float): Temperature in kelvin, if not specified
                will return non-cv-normalized value
            structure (Structure): Structure to be used in directional heat
                capacity determination, only necessary if temperature
                is specified
            quadct (dict): quadrature for integration, should be
                dictionary with "points" and "weights" keys defaults
                to quadpy.sphere.Lebedev(19) as read from file
        """
    if temperature and (not structure):
        raise ValueError('If using temperature input, you must also include structure')
    quad = quad or DEFAULT_QUAD
    points = quad['points']
    weights = quad['weights']
    num, denom, c = (np.zeros((3, 3)), 0, 1)
    for p, w in zip(points, weights):
        gk = ElasticTensor(self[0]).green_kristoffel(p)
        _rho_wsquareds, us = np.linalg.eigh(gk)
        us = [u / np.linalg.norm(u) for u in np.transpose(us)]
        for u in us:
            if temperature and structure:
                c = self.get_heat_capacity(temperature, structure, p, u)
            num += c * self.get_ggt(p, u) * w
            denom += c * w
    return SquareTensor(num / denom)