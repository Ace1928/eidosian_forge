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
def get_all_nn_info(self, structure: Structure) -> list[list[dict[str, Any]]]:
    """
        Args:
            structure (Structure): input structure.

        Returns:
            List of near neighbor information for each site. See get_nn_info for the
            format of the data for each site.
        """
    all_nns = self.get_all_voronoi_polyhedra(structure)
    return [self._filter_nns(structure, n, nns) for n, nns in enumerate(all_nns)]