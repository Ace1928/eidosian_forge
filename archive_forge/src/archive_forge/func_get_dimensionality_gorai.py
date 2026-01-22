from __future__ import annotations
import copy
import itertools
from collections import defaultdict
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.structure_analyzer import get_max_bond_lengths
from pymatgen.core import Molecule, Species, Structure
from pymatgen.core.lattice import get_integer_index
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_dimensionality_gorai(structure, max_hkl=2, el_radius_updates=None, min_slab_size=5, min_vacuum_size=5, standardize=True, bonds=None):
    """
    This method returns whether a structure is 3D, 2D (layered), or 1D (linear
    chains or molecules) according to the algorithm published in Gorai, P.,
    Toberer, E. & Stevanovic, V. Computational Identification of Promising
    Thermoelectric Materials Among Known Quasi-2D Binary Compounds. J. Mater.
    Chem. A 2, 4136 (2016).

    Note that a 1D structure detection might indicate problems in the bonding
    algorithm, particularly for ionic crystals (e.g., NaCl)

    Users can change the behavior of bonds detection by passing either
    el_radius_updates to update atomic radii for auto-detection of max bond
    distances, or bonds to explicitly specify max bond distances for atom pairs.
    Note that if you pass both, el_radius_updates are ignored.

    Args:
        structure (Structure): structure to analyze dimensionality for
        max_hkl (int): max index of planes to look for layers
        el_radius_updates (dict): symbol->float to update atomic radii
        min_slab_size (float): internal surface construction parameter
        min_vacuum_size (float): internal surface construction parameter
        standardize (bool): whether to standardize the structure before
            analysis. Set to False only if you already have the structure in a
            convention where layers / chains will be along low <hkl> indexes.
        bonds (dict[tuple, float]): bonds are specified as a dict of 2-tuples of Species mapped to floats,
            the max bonding distance. For example, PO4 groups may be
            defined as {("P", "O"): 3}.

    Returns:
        int: the dimensionality of the structure - 1 (molecules/chains),
            2 (layered), or 3 (3D)
    """
    if standardize:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    if not bonds:
        bonds = get_max_bond_lengths(structure, el_radius_updates)
    n_surfaces = 0
    for hh in range(max_hkl):
        for kk in range(max_hkl):
            for ll in range(max_hkl):
                if max([hh, kk, ll]) > 0 and n_surfaces < 2:
                    sg = SlabGenerator(structure, (hh, kk, ll), min_slab_size=min_slab_size, min_vacuum_size=min_vacuum_size)
                    slabs = sg.get_slabs(bonds)
                    for _ in slabs:
                        n_surfaces += 1
    return 3 - min(n_surfaces, 2)