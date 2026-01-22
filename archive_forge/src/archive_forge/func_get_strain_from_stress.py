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
def get_strain_from_stress(self, stress):
    """
        Gets the strain from a stress state according
        to the compliance expansion corresponding to the
        tensor expansion.
        """
    compl_exp = self.get_compliance_expansion()
    strain = 0
    for n, compl in enumerate(compl_exp, start=1):
        strain += compl.einsum_sequence([stress] * n) / factorial(n)
    return strain