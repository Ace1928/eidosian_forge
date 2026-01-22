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
def thermal_expansion_coeff(self, structure: Structure, temperature: float, mode: Literal['dulong - petit', 'debye']='debye'):
    """
        Gets thermal expansion coefficient from third-order constants.

        Args:
            temperature (float): Temperature in kelvin, if not specified
                will return non-cv-normalized value
            structure (Structure): Structure to be used in directional heat
                capacity determination, only necessary if temperature
                is specified
            mode (str): mode for finding average heat-capacity,
                current supported modes are 'debye' and 'dulong-petit'
        """
    soec = ElasticTensor(self[0])
    v0 = structure.volume * 1e-30 / len(structure)
    if mode == 'debye':
        td = soec.debye_temperature(structure)
        t_ratio = temperature / td

        def integrand(x):
            return x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2
        cv = 9 * 8.314 * t_ratio ** 3 * quad(integrand, 0, t_ratio ** (-1))[0]
    elif mode == 'dulong-petit':
        cv = 3 * 8.314
    else:
        raise ValueError(f'mode={mode!r} must be debye or dulong-petit')
    tgt = self.get_tgt(temperature, structure)
    alpha = np.einsum('ijkl,ij', soec.compliance_tensor, tgt)
    alpha *= cv / (1000000000.0 * v0 * 6.022e+23)
    return SquareTensor(alpha)