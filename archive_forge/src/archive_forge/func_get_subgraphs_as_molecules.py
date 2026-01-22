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
def get_subgraphs_as_molecules(self, use_weights: bool=False) -> list[Molecule]:
    """
        Retrieve subgraphs as molecules, useful for extracting
        molecules from periodic crystals.

        Will only return unique molecules, not any duplicates
        present in the crystal (a duplicate defined as an
        isomorphic subgraph).

        Args:
            use_weights (bool): If True, only treat subgraphs
                as isomorphic if edges have the same weights. Typically,
                this means molecules will need to have the same bond
                lengths to be defined as duplicates, otherwise bond
                lengths can differ. This is a fairly robust approach,
                but will treat e.g. enantiomers as being duplicates.

        Returns:
            list of unique Molecules in Structure
        """
    if getattr(self, '_supercell_sg', None) is None:
        self._supercell_sg = supercell_sg = self * (3, 3, 3)
    supercell_sg.graph = nx.Graph(supercell_sg.graph)
    all_subgraphs = [supercell_sg.graph.subgraph(c) for c in nx.connected_components(supercell_sg.graph)]
    molecule_subgraphs = []
    for subgraph in all_subgraphs:
        intersects_boundary = any((d['to_jimage'] != (0, 0, 0) for u, v, d in subgraph.edges(data=True)))
        if not intersects_boundary:
            molecule_subgraphs.append(nx.MultiDiGraph(subgraph))
    for subgraph in molecule_subgraphs:
        for n in subgraph:
            subgraph.add_node(n, specie=str(supercell_sg.structure[n].specie))

    def node_match(n1, n2):
        return n1['specie'] == n2['specie']

    def edge_match(e1, e2):
        if use_weights:
            return e1['weight'] == e2['weight']
        return True
    unique_subgraphs: list = []
    for subgraph in molecule_subgraphs:
        already_present = [nx.is_isomorphic(subgraph, g, node_match=node_match, edge_match=edge_match) for g in unique_subgraphs]
        if not any(already_present):
            unique_subgraphs.append(subgraph)
    molecules = []
    for subgraph in unique_subgraphs:
        coords = [supercell_sg.structure[n].coords for n in subgraph.nodes()]
        species = [supercell_sg.structure[n].specie for n in subgraph.nodes()]
        molecule = Molecule(species, coords)
        molecule = molecule.get_centered_molecule()
        molecules.append(molecule)
    return molecules