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
def get_max_bond_distance(self, el1_sym, el2_sym):
    """
        Use Jmol algorithm to determine bond length from atomic parameters

        Args:
            el1_sym (str): symbol of atom 1
            el2_sym (str): symbol of atom 2.

        Returns:
            float: max bond length
        """
    return sqrt((self.el_radius[el1_sym] + self.el_radius[el2_sym] + self.tol) ** 2)