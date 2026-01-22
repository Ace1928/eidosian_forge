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
def get_structure_components(bonded_structure, inc_orientation=False, inc_site_ids=False, inc_molecule_graph=False):
    """
    Gets information on the components in a bonded structure.

    Correctly determines the dimensionality of all structures, regardless of
    structure type or improper connections due to periodic boundary conditions.

    Requires a StructureGraph object as input. This can be generated using one
    of the NearNeighbor classes. For example, using the CrystalNN class:

        bonded_structure = CrystalNN().get_bonded_structure(structure)

    Based on the modified breadth-first-search algorithm described in:

    P. M. Larsen, M. Pandey, M. Strange, K. W. Jacobsen. Definition of a
    scoring parameter to identify low-dimensional materials components.
    Phys. Rev. Materials 3, 034003 (2019).

    Args:
        bonded_structure (StructureGraph): A structure with bonds, represented
            as a pymatgen structure graph. For example, generated using the
            CrystalNN.get_bonded_structure() method.
        inc_orientation (bool, optional): Whether to include the orientation
            of the structure component. For surfaces, the miller index is given,
            for one-dimensional structures, the direction of the chain is given.
        inc_site_ids (bool, optional): Whether to include the site indices
            of the sites in the structure component.
        inc_molecule_graph (bool, optional): Whether to include MoleculeGraph
            objects for zero-dimensional components.

    Returns:
        list[dict]: Information on the components in a structure as a list
            of dictionaries with the keys:

            - "structure_graph": A pymatgen StructureGraph object for the
                component.
            - "dimensionality": The dimensionality of the structure component as an
                int.
            - "orientation": If inc_orientation is `True`, the orientation of the
                component as a tuple. E.g. (1, 1, 1)
            - "site_ids": If inc_site_ids is `True`, the site indices of the
                sites in the component as a tuple.
            - "molecule_graph": If inc_molecule_graph is `True`, the site a
                MoleculeGraph object for zero-dimensional components.
    """
    comp_graphs = (bonded_structure.graph.subgraph(c) for c in nx.weakly_connected_components(bonded_structure.graph))
    components = []
    for graph in comp_graphs:
        dimensionality, vertices = calculate_dimensionality_of_site(bonded_structure, next(iter(graph.nodes())), inc_vertices=True)
        component = {'dimensionality': dimensionality}
        if inc_orientation:
            if dimensionality in [1, 2]:
                vertices = np.array(vertices)
                g = vertices.sum(axis=0) / vertices.shape[0]
                _, _, vh = np.linalg.svd(vertices - g)
                index = 2 if dimensionality == 2 else 0
                orientation = get_integer_index(vh[index, :])
            else:
                orientation = None
            component['orientation'] = orientation
        if inc_site_ids:
            component['site_ids'] = tuple(graph.nodes())
        if inc_molecule_graph and dimensionality == 0:
            component['molecule_graph'] = zero_d_graph_to_molecule_graph(bonded_structure, graph)
        component_structure = Structure.from_sites([bonded_structure.structure[n] for n in sorted(graph.nodes())])
        sorted_graph = nx.convert_node_labels_to_integers(graph, ordering='sorted')
        component_graph = StructureGraph(component_structure, graph_data=json_graph.adjacency_data(sorted_graph))
        component['structure_graph'] = component_graph
        components.append(component)
    return components