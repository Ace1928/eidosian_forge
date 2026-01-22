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
def get_cn_dict(self, structure: Structure, n: int, use_weights: bool=False, **kwargs):
    """
        Get coordination number, CN, of each element bonded to site with index n in structure.

        Args:
            structure (Structure): input structure
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).

        Returns:
            cn (dict): dictionary of CN of each element bonded to site
        """
    if self.weighted_cn != use_weights:
        raise ValueError('The weighted_cn parameter and use_weights parameter should match!')
    return super().get_cn_dict(structure, n, use_weights)