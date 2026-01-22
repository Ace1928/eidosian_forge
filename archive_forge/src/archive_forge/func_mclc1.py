from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
def mclc1(self, a, b, c, alpha):
    """MCLC1 Path."""
    self.name = 'MCLC1'
    zeta = (2 - b * cos(alpha) / c) / (4 * sin(alpha) ** 2)
    eta = 0.5 + 2 * zeta * c * cos(alpha) / b
    psi = 0.75 - a ** 2 / (4 * b ** 2 * sin(alpha) ** 2)
    phi = psi + (0.75 - psi) * b * cos(alpha) / c
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'N': np.array([0.5, 0.0, 0.0]), 'N_1': np.array([0.0, -0.5, 0.0]), 'F': np.array([1 - zeta, 1 - zeta, 1 - eta]), 'F_1': np.array([zeta, zeta, eta]), 'F_2': np.array([-zeta, -zeta, 1 - eta]), 'I': np.array([phi, 1 - phi, 0.5]), 'I_1': np.array([1 - phi, phi - 1, 0.5]), 'L': np.array([0.5, 0.5, 0.5]), 'M': np.array([0.5, 0.0, 0.5]), 'X': np.array([1 - psi, psi - 1, 0.0]), 'X_1': np.array([psi, 1 - psi, 0.0]), 'X_2': np.array([psi - 1, -psi, 0.0]), 'Y': np.array([0.5, 0.5, 0.0]), 'Y_1': np.array([-0.5, -0.5, 0.0]), 'Z': np.array([0.0, 0.0, 0.5])}
    path = [['\\Gamma', 'Y', 'F', 'L', 'I'], ['I_1', 'Z', 'F_1'], ['Y', 'X_1'], ['X', '\\Gamma', 'N'], ['M', '\\Gamma']]
    return {'kpoints': kpoints, 'path': path}