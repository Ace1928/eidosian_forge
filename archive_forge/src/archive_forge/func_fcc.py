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
def fcc(self):
    """FCC Path."""
    self.name = 'FCC'
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'K': np.array([3.0 / 8.0, 3.0 / 8.0, 3.0 / 4.0]), 'L': np.array([0.5, 0.5, 0.5]), 'U': np.array([5.0 / 8.0, 1.0 / 4.0, 5.0 / 8.0]), 'W': np.array([0.5, 1.0 / 4.0, 3.0 / 4.0]), 'X': np.array([0.5, 0.0, 0.5])}
    path = [['\\Gamma', 'X', 'W', 'K', '\\Gamma', 'L', 'U', 'W', 'L', 'K'], ['U', 'X']]
    return {'kpoints': kpoints, 'path': path}