"""Algorithms Lab: reusable, high-performance simulation building blocks."""

from algorithms_lab.backends import HAS_NUMBA, HAS_SCIPY
from algorithms_lab.gpu import CuPyNBody, OpenCLNBody, HAS_CUPY, HAS_PYOPENCL
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.spatial_hash import UniformGrid
from algorithms_lab.neighbor_list import NeighborList
from algorithms_lab.neighbors import NeighborSearch
from algorithms_lab.kdtree import KDTreeNeighborSearch
from algorithms_lab.morton import morton_encode, morton_sort
from algorithms_lab.barnes_hut import BarnesHutTree
from algorithms_lab.fmm2d import FMM2D
from algorithms_lab.fmm_multilevel import MultiLevelFMM
from algorithms_lab.sph import SPHSolver
from algorithms_lab.pbf import PBFSolver
from algorithms_lab.xpbd import XPBFSolver
from algorithms_lab.graph import NeighborGraph, build_neighbor_graph
from algorithms_lab.forces import ForceDefinition, ForcePack, ForceRegistry, ForceType
from algorithms_lab.spatial_utils import (
    GridConfig,
    adaptive_cell_size,
    compute_batch_ranges,
    compute_cell_densities,
    compute_morton_order,
    morton_decode_2d,
    morton_encode_2d,
    pack_positions_soa,
    prefetch_neighbor_data,
    unpack_positions_aos,
)

__all__ = [
    "Domain",
    "WrapMode",
    "HAS_NUMBA",
    "HAS_SCIPY",
    "HAS_CUPY",
    "HAS_PYOPENCL",
    "UniformGrid",
    "NeighborList",
    "NeighborSearch",
    "KDTreeNeighborSearch",
    "morton_encode",
    "morton_sort",
    "BarnesHutTree",
    "FMM2D",
    "MultiLevelFMM",
    "SPHSolver",
    "PBFSolver",
    "XPBFSolver",
    "NeighborGraph",
    "build_neighbor_graph",
    "ForceDefinition",
    "ForcePack",
    "ForceRegistry",
    "ForceType",
    "GridConfig",
    "adaptive_cell_size",
    "compute_batch_ranges",
    "compute_cell_densities",
    "compute_morton_order",
    "morton_decode_2d",
    "morton_encode_2d",
    "pack_positions_soa",
    "prefetch_neighbor_data",
    "unpack_positions_aos",
    "CuPyNBody",
    "OpenCLNBody",
]
