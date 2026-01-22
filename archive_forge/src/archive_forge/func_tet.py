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
def tet(self):
    """TET Path."""
    self.name = 'TET'
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'A': np.array([0.5, 0.5, 0.5]), 'M': np.array([0.5, 0.5, 0.0]), 'R': np.array([0.0, 0.5, 0.5]), 'X': np.array([0.0, 0.5, 0.0]), 'Z': np.array([0.0, 0.0, 0.5])}
    path = [['\\Gamma', 'X', 'M', '\\Gamma', 'Z', 'R', 'A', 'Z'], ['X', 'R'], ['M', 'A']]
    return {'kpoints': kpoints, 'path': path}