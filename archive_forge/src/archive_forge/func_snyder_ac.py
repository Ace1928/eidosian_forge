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
@raise_if_unphysical
def snyder_ac(self, structure: Structure) -> float:
    """Calculates Snyder's acoustic sound velocity.

        Args:
            structure: pymatgen structure object

        Returns:
            float: Snyder's acoustic sound velocity (in SI units)
        """
    n_sites = len(structure)
    n_atoms = structure.composition.num_atoms
    site_density = 1e+30 * n_sites / structure.volume
    tot_mass = sum((spec.atomic_mass for spec in structure.species))
    avg_mass = 1.6605e-27 * tot_mass / n_atoms
    return 0.38483 * avg_mass * ((self.long_v(structure) + 2 * self.trans_v(structure)) / 3) ** 3.0 / (300 * site_density ** (-2 / 3) * n_sites ** (1 / 3))