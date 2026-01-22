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
def mclc3(self, a, b, c, alpha):
    """MCLC3 Path."""
    self.name = 'MCLC3'
    mu = (1 + b ** 2 / a ** 2) / 4.0
    delta = b * c * cos(alpha) / (2 * a ** 2)
    zeta = mu - 0.25 + (1 - b * cos(alpha) / c) / (4 * sin(alpha) ** 2)
    eta = 0.5 + 2 * zeta * c * cos(alpha) / b
    phi = 1 + zeta - 2 * mu
    psi = eta - 2 * delta
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'F': np.array([1 - phi, 1 - phi, 1 - psi]), 'F_1': np.array([phi, phi - 1, psi]), 'F_2': np.array([1 - phi, -phi, 1 - psi]), 'H': np.array([zeta, zeta, eta]), 'H_1': np.array([1 - zeta, -zeta, 1 - eta]), 'H_2': np.array([-zeta, -zeta, 1 - eta]), 'I': np.array([0.5, -0.5, 0.5]), 'M': np.array([0.5, 0.0, 0.5]), 'N': np.array([0.5, 0.0, 0.0]), 'N_1': np.array([0.0, -0.5, 0.0]), 'X': np.array([0.5, -0.5, 0.0]), 'Y': np.array([mu, mu, delta]), 'Y_1': np.array([1 - mu, -mu, -delta]), 'Y_2': np.array([-mu, -mu, -delta]), 'Y_3': np.array([mu, mu - 1, delta]), 'Z': np.array([0.0, 0.0, 0.5])}
    path = [['\\Gamma', 'Y', 'F', 'H', 'Z', 'I', 'F_1'], ['H_1', 'Y_1', 'X', '\\Gamma', 'N'], ['M', '\\Gamma']]
    return {'kpoints': kpoints, 'path': path}