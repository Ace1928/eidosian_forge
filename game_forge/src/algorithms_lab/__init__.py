"""Algorithms Lab: reusable, high-performance simulation building blocks."""

from algorithms_lab.backends import HAS_NUMBA, HAS_SCIPY
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.spatial_hash import UniformGrid
from algorithms_lab.neighbor_list import NeighborList
from algorithms_lab.neighbors import NeighborSearch
from algorithms_lab.kdtree import KDTreeNeighborSearch
from algorithms_lab.morton import morton_encode, morton_sort
from algorithms_lab.barnes_hut import BarnesHutTree
from algorithms_lab.fmm2d import FMM2D
from algorithms_lab.sph import SPHSolver
from algorithms_lab.pbf import PBFSolver
from algorithms_lab.xpbd import XPBFSolver

__all__ = [
    "Domain",
    "WrapMode",
    "HAS_NUMBA",
    "HAS_SCIPY",
    "UniformGrid",
    "NeighborList",
    "NeighborSearch",
    "KDTreeNeighborSearch",
    "morton_encode",
    "morton_sort",
    "BarnesHutTree",
    "FMM2D",
    "SPHSolver",
    "PBFSolver",
    "XPBFSolver",
]
