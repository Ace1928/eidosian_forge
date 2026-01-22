from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
def get_neighbors_of_site_with_index(struct, n, approach='min_dist', delta=0.1, cutoff=10):
    """
    Returns the neighbors of a given site using a specific neighbor-finding
    method.

    Args:
        struct (Structure): input structure.
        n (int): index of site in Structure object for which motif type
            is to be determined.
        approach (str): type of neighbor-finding approach, where
            "min_dist" will use the MinimumDistanceNN class,
            "voronoi" the VoronoiNN class, "min_OKeeffe" the
            MinimumOKeeffe class, and "min_VIRE" the MinimumVIRENN class.
        delta (float): tolerance involved in neighbor finding.
        cutoff (float): radius to find tentative neighbors.

    Returns:
        neighbor sites.
    """
    if approach == 'min_dist':
        return MinimumDistanceNN(tol=delta, cutoff=cutoff).get_nn(struct, n)
    if approach == 'voronoi':
        return VoronoiNN(tol=delta, cutoff=cutoff).get_nn(struct, n)
    if approach == 'min_OKeeffe':
        return MinimumOKeeffeNN(tol=delta, cutoff=cutoff).get_nn(struct, n)
    if approach == 'min_VIRE':
        return MinimumVIRENN(tol=delta, cutoff=cutoff).get_nn(struct, n)
    raise RuntimeError(f'unsupported neighbor-finding method ({approach}).')