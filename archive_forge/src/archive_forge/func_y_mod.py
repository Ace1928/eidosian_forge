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
@property
def y_mod(self) -> float:
    """
        Calculates Young's modulus (in SI units) using the
        Voigt-Reuss-Hill averages of bulk and shear moduli.
        """
    return 9000000000.0 * self.k_vrh * self.g_vrh / (3 * self.k_vrh + self.g_vrh)