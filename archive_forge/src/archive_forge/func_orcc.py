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
def orcc(self, a, b, c):
    """ORCC Path."""
    self.name = 'ORCC'
    zeta = (1 + a ** 2 / b ** 2) / 4
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'A': np.array([zeta, zeta, 0.5]), 'A_1': np.array([-zeta, 1 - zeta, 0.5]), 'R': np.array([0.0, 0.5, 0.5]), 'S': np.array([0.0, 0.5, 0.0]), 'T': np.array([-0.5, 0.5, 0.5]), 'X': np.array([zeta, zeta, 0.0]), 'X_1': np.array([-zeta, 1 - zeta, 0.0]), 'Y': np.array([-0.5, 0.5, 0]), 'Z': np.array([0.0, 0.0, 0.5])}
    path = [['\\Gamma', 'X', 'S', 'R', 'A', 'Z', '\\Gamma', 'Y', 'X_1', 'A_1', 'T', 'Y'], ['Z', 'T']]
    return {'kpoints': kpoints, 'path': path}