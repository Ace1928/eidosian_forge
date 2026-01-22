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
def get_yield_stress(self, n):
    """
        Gets the yield stress for a given direction.

        Args:
            n (3x1 array-like): direction for which to find the
                yield stress
        """
    comp = root(self.get_stability_criteria, -1, args=n)
    tens = root(self.get_stability_criteria, 1, args=n)
    return (comp.x, tens.x)