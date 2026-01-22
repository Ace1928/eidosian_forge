from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def check_for_structure_changes(mol1: Molecule, mol2: Molecule) -> str:
    """
    Compares connectivity of two molecules (using MoleculeGraph w/ OpenBabelNN).
    This function will work with two molecules with different atom orderings,
        but for proper treatment, atoms should be listed in the same order.
    Possible outputs include:
    - no_change: the bonding in the two molecules is identical
    - unconnected_fragments: the MoleculeGraph of mol1 is connected, but the
      MoleculeGraph is mol2 is not connected
    - fewer_bonds: the MoleculeGraph of mol1 has more bonds (edges) than the
      MoleculeGraph of mol2
    - more_bonds: the MoleculeGraph of mol2 has more bonds (edges) than the
      MoleculeGraph of mol1
    - bond_change: this case catches any other non-identical MoleculeGraphs
    Args:
        mol1: Pymatgen Molecule object to be compared.
        mol2: Pymatgen Molecule object to be compared.

    Returns:
        One of ["unconnected_fragments", "fewer_bonds", "more_bonds",
        "bond_change", "no_change"]
    """
    special_elements = ['Li', 'Na', 'Mg', 'Ca', 'Zn']
    mol_list = [copy.deepcopy(mol1), copy.deepcopy(mol2)]
    if mol1.composition != mol2.composition:
        raise RuntimeError('Molecules have different compositions! Exiting...')
    for ii, site in enumerate(mol1):
        if site.specie.symbol != mol2[ii].specie.symbol:
            warnings.warn('Comparing molecules with different atom ordering! Turning off special treatment for coordinating metals.')
            special_elements = []
    special_sites: list[list] = [[], []]
    for ii, mol in enumerate(mol_list):
        for jj, site in enumerate(mol):
            if site.specie.symbol in special_elements:
                distances = [[kk, site.distance(other_site)] for kk, other_site in enumerate(mol)]
                special_sites[ii].append([jj, site, distances])
        for jj, site in enumerate(mol):
            if site.specie.symbol in special_elements:
                del mol[jj]
    initial_mol_graph = MoleculeGraph.from_local_env_strategy(mol_list[0], OpenBabelNN())
    initial_graph = initial_mol_graph.graph
    last_mol_graph = MoleculeGraph.from_local_env_strategy(mol_list[1], OpenBabelNN())
    last_graph = last_mol_graph.graph
    if initial_mol_graph.isomorphic_to(last_mol_graph):
        return 'no_change'
    if nx.is_connected(initial_graph.to_undirected()) and (not nx.is_connected(last_graph.to_undirected())):
        return 'unconnected_fragments'
    if last_graph.number_of_edges() < initial_graph.number_of_edges():
        return 'fewer_bonds'
    if last_graph.number_of_edges() > initial_graph.number_of_edges():
        return 'more_bonds'
    return 'bond_change'