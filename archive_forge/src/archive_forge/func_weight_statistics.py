from __future__ import annotations
import copy
import logging
import os.path
import subprocess
import warnings
from collections import defaultdict, namedtuple
from itertools import combinations
from operator import itemgetter
from shutil import which
from typing import TYPE_CHECKING, Any, Callable, cast
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from networkx.drawing.nx_agraph import write_dot
from networkx.readwrite import json_graph
from scipy.spatial import KDTree
from scipy.stats import describe
from pymatgen.core import Lattice, Molecule, PeriodicSite, Structure
from pymatgen.core.structure import FunctionalGroups
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.vis.structure_vtk import EL_COLORS
@property
def weight_statistics(self) -> dict:
    """
        Extract a statistical summary of edge weights present in
        the graph.

        Returns:
            A dict with an 'all_weights' list, 'minimum',
            'maximum', 'median', 'mean', 'std_dev'
        """
    all_weights = [d.get('weight') for u, v, d in self.graph.edges(data=True)]
    stats = describe(all_weights, nan_policy='omit')
    return {'all_weights': all_weights, 'min': stats.minmax[0], 'max': stats.minmax[1], 'mean': stats.mean, 'variance': stats.variance}