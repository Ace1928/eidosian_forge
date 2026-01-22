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
def substitute_group(self, index, func_grp, strategy, bond_order=1, graph_dict=None, strategy_params=None):
    """
        Builds off of Molecule.substitute to replace an atom in self.molecule
        with a functional group. This method also amends self.graph to
        incorporate the new functional group.

        NOTE: using a MoleculeGraph will generally produce a different graph
        compared with using a Molecule or str (when not using graph_dict).

        Args:
            index: Index of atom to substitute.
            func_grp: Substituent molecule. There are three options:
                1. Providing an actual molecule as the input. The first atom
                    must be a DummySpecies X, indicating the position of
                    nearest neighbor. The second atom must be the next
                    nearest atom. For example, for a methyl group
                    substitution, func_grp should be X-CH3, where X is the
                    first site and C is the second site. What the code will
                    do is to remove the index site, and connect the nearest
                    neighbor to the C atom in CH3. The X-C bond indicates the
                    directionality to connect the atoms.
                2. A string name. The molecule will be obtained from the
                    relevant template in func_groups.json.
                3. A MoleculeGraph object.
            strategy: Class from pymatgen.analysis.local_env.
            bond_order: A specified bond order to calculate the bond
                length between the attached functional group and the nearest
                neighbor site. Defaults to 1.
            graph_dict: Dictionary representing the bonds of the functional
                group (format: {(u, v): props}, where props is a dictionary of
                properties, including weight. If None, then the algorithm
                will attempt to automatically determine bonds using one of
                a list of strategies defined in pymatgen.analysis.local_env.
            strategy_params: dictionary of keyword arguments for strategy.
                If None, default parameters will be used.
        """

    def map_indices(grp):
        grp_map = {}
        atoms = len(grp) - 1
        offset = len(self.molecule) - atoms
        for idx in range(atoms):
            grp_map[idx] = idx + offset
        return grp_map
    if isinstance(func_grp, MoleculeGraph):
        self.molecule.substitute(index, func_grp.molecule, bond_order=bond_order)
        mapping = map_indices(func_grp.molecule)
        for u, v in list(func_grp.graph.edges()):
            edge_props = func_grp.graph.get_edge_data(u, v)[0]
            weight = edge_props.pop('weight', None)
            self.add_edge(mapping[u], mapping[v], weight=weight, edge_properties=edge_props)
    else:
        if isinstance(func_grp, Molecule):
            func_grp = copy.deepcopy(func_grp)
        else:
            try:
                func_grp = copy.deepcopy(FunctionalGroups[func_grp])
            except Exception:
                raise RuntimeError("Can't find functional group in list. Provide explicit coordinate instead")
        self.molecule.substitute(index, func_grp, bond_order=bond_order)
        mapping = map_indices(func_grp)
        func_grp.remove_species('X')
        if graph_dict is not None:
            for u, v in graph_dict:
                edge_props = graph_dict[u, v]
                weight = edge_props.pop('weight', None)
                self.add_edge(mapping[u], mapping[v], weight=weight, edge_properties=edge_props)
        else:
            graph = self.from_local_env_strategy(func_grp, strategy(**strategy_params or {}))
            for u, v in list(graph.graph.edges()):
                edge_props = graph.graph.get_edge_data(u, v)[0]
                weight = edge_props.pop('weight', None)
                if 0 not in list(graph.graph.nodes()):
                    u, v = (u - 1, v - 1)
                self.add_edge(mapping[u], mapping[v], weight=weight, edge_properties=edge_props)