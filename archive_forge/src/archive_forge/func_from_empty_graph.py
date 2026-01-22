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
def from_empty_graph(cls, molecule, name='bonds', edge_weight_name=None, edge_weight_units=None) -> Self:
    """
        Constructor for MoleculeGraph, returns a MoleculeGraph
        object with an empty graph (no edges, only nodes defined
        that correspond to Sites in Molecule).

        Args:
            molecule (Molecule):
            name (str): name of graph, e.g. "bonds"
            edge_weight_name (str): name of edge weights,
                e.g. "bond_length" or "exchange_constant"
            edge_weight_units (str): name of edge weight units
            e.g. "Å" or "eV"

        Returns:
            MoleculeGraph
        """
    if edge_weight_name and edge_weight_units is None:
        raise ValueError('Please specify units associated with your edge weights. Can be empty string if arbitrary or dimensionless.')
    graph = nx.MultiDiGraph(edge_weight_name=edge_weight_name, edge_weight_units=edge_weight_units, name=name)
    graph.add_nodes_from(range(len(molecule)))
    graph_data = json_graph.adjacency_data(graph)
    return cls(molecule, graph_data=graph_data)