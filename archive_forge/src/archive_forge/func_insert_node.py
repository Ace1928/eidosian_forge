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
def insert_node(self, idx, species, coords, validate_proximity=False, site_properties=None, edges=None):
    """
        A wrapper around Molecule.insert(), which also incorporates the new
        site into the MoleculeGraph.

        Args:
            idx: Index at which to insert the new site
            species: Species for the new site
            coords: 3x1 array representing coordinates of the new site
            validate_proximity: For Molecule.insert(); if True (default
                False), distance will be checked to ensure that site can be safely
                added.
            site_properties: Site properties for Molecule
            edges: List of dicts representing edges to be added to the
                MoleculeGraph. These edges must include the index of the new site i,
                and all indices used for these edges should reflect the
                MoleculeGraph AFTER the insertion, NOT before. Each dict should at
                least have a "to_index" and "from_index" key, and can also have a
                "weight" and a "properties" key.
        """
    self.molecule.insert(idx, species, coords, validate_proximity=validate_proximity, properties=site_properties)
    mapping = {}
    for j in range(len(self.molecule) - 1):
        if j < idx:
            mapping[j] = j
        else:
            mapping[j] = j + 1
    nx.relabel_nodes(self.graph, mapping, copy=False)
    self.graph.add_node(idx)
    self.set_node_attributes()
    if edges is not None:
        for edge in edges:
            try:
                self.add_edge(edge['from_index'], edge['to_index'], weight=edge.get('weight'), edge_properties=edge.get('properties'))
            except KeyError:
                raise RuntimeError('Some edges are invalid.')