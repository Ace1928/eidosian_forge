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
@classmethod
def from_local_env_strategy(cls, molecule, strategy) -> Self:
    """
        Constructor for MoleculeGraph, using a strategy
        from pymatgen.analysis.local_env.

            molecule: Molecule object
            strategy: an instance of a
                pymatgen.analysis.local_env.NearNeighbors object

        Returns:
            mg, a MoleculeGraph
        """
    if not strategy.molecules_allowed:
        raise ValueError(f'strategy={strategy!r} is not designed for use with molecules! Choose another strategy.')
    extend_structure = strategy.extend_structure_molecules
    mg = cls.from_empty_graph(molecule, name='bonds', edge_weight_name='weight', edge_weight_units='')
    coords = molecule.cart_coords
    if extend_structure:
        a = max(coords[:, 0]) - min(coords[:, 0]) + 100
        b = max(coords[:, 1]) - min(coords[:, 1]) + 100
        c = max(coords[:, 2]) - min(coords[:, 2]) + 100
        structure = molecule.get_boxed_structure(a, b, c, no_cross=True, reorder=False)
    else:
        structure = None
    for idx in range(len(molecule)):
        neighbors = strategy.get_nn_info(molecule, idx) if structure is None else strategy.get_nn_info(structure, idx)
        for neighbor in neighbors:
            if not np.array_equal(neighbor['image'], [0, 0, 0]):
                continue
            if idx > neighbor['site_index']:
                from_index = neighbor['site_index']
                to_index = idx
            else:
                from_index = idx
                to_index = neighbor['site_index']
            mg.add_edge(from_index=from_index, to_index=to_index, weight=neighbor['weight'], warn_duplicates=False)
    duplicates = []
    for edge in mg.graph.edges:
        if edge[2] != 0:
            duplicates.append(edge)
    for duplicate in duplicates:
        mg.graph.remove_edge(duplicate[0], duplicate[1], key=duplicate[2])
    mg.set_node_attributes()
    return mg