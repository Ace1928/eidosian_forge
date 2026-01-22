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
def get_bonded_structure(self, structure: Structure, decorate: bool=False) -> StructureGraph:
    """
        Args:
            structure (Structure): Input structure
            decorate (bool, optional): Whether to decorate the structure. Defaults to False.

        Returns:
            StructureGraph: Bonded structure
        """
    from pymatgen.command_line.critic2_caller import Critic2Caller
    if structure == self._last_structure:
        sg = self._last_bonded_structure
    else:
        c2_output = Critic2Caller(structure).output
        sg = c2_output.structure_graph()
        self._last_structure = structure
        self._last_bonded_structure = sg
    if decorate:
        order_parameters = [self.get_local_order_parameters(structure, n) for n in range(len(structure))]
        sg.structure.add_site_property('order_parameters', order_parameters)
    return sg