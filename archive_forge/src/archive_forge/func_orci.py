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
def orci(self, a, b, c):
    """ORCI Path."""
    self.name = 'ORCI'
    zeta = (1 + a ** 2 / c ** 2) / 4
    eta = (1 + b ** 2 / c ** 2) / 4
    delta = (b ** 2 - a ** 2) / (4 * c ** 2)
    mu = (a ** 2 + b ** 2) / (4 * c ** 2)
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'L': np.array([-mu, mu, 0.5 - delta]), 'L_1': np.array([mu, -mu, 0.5 + delta]), 'L_2': np.array([0.5 - delta, 0.5 + delta, -mu]), 'R': np.array([0.0, 0.5, 0.0]), 'S': np.array([0.5, 0.0, 0.0]), 'T': np.array([0.0, 0.0, 0.5]), 'W': np.array([0.25, 0.25, 0.25]), 'X': np.array([-zeta, zeta, zeta]), 'X_1': np.array([zeta, 1 - zeta, -zeta]), 'Y': np.array([eta, -eta, eta]), 'Y_1': np.array([1 - eta, eta, -eta]), 'Z': np.array([0.5, 0.5, -0.5])}
    path = [['\\Gamma', 'X', 'L', 'T', 'W', 'R', 'X_1', 'Z', '\\Gamma', 'Y', 'S', 'W'], ['L_1', 'Y'], ['Y_1', 'Z']]
    return {'kpoints': kpoints, 'path': path}