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
def get_local_order_parameters(self, structure: Structure, n: int):
    """
        Calculate those local structure order parameters for
        the given site whose ideal CN corresponds to the
        underlying motif (e.g., CN=4, then calculate the
        square planar, tetrahedral, see-saw-like,
        rectangular see-saw-like order parameters).

        Args:
            structure: Structure object
            n (int): site index.

        Returns:
            dict[str, float]: A dict of order parameters (values) and the
                underlying motif type (keys; for example, tetrahedral).
        """
    cn = self.get_cn(structure, n)
    int_cn = [int(k_cn) for k_cn in cn_opt_params]
    if cn in int_cn:
        names = list(cn_opt_params[cn])
        types = []
        params = []
        for name in names:
            types.append(cn_opt_params[cn][name][0])
            tmp = cn_opt_params[cn][name][1] if len(cn_opt_params[cn][name]) > 1 else None
            params.append(tmp)
        lsops = LocalStructOrderParams(types, parameters=params)
        sites = [structure[n], *self.get_nn(structure, n)]
        lostop_vals = lsops.get_order_parameters(sites, 0, indices_neighs=list(range(1, cn + 1)))
        dct = {}
        for idx, lsop in enumerate(lostop_vals):
            dct[names[idx]] = lsop
        return dct
    return None