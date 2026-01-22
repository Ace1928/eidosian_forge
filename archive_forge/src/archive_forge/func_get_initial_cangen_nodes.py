import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def get_initial_cangen_nodes(subgraph, enumeration_mol, atom_assignment, do_initial_assignment=True):
    atom_map = {}
    cangen_nodes = []
    atoms = enumeration_mol.atoms
    canonical_labels = []
    for i, atom_index in enumerate(subgraph.atom_indices):
        atom_map[atom_index] = i
        cangen_nodes.append(CangenNode(i, atoms[atom_index].atom_smarts))
        canonical_labels.append([])
    for bond_index in subgraph.bond_indices:
        bond = enumeration_mol.bonds[bond_index]
        from_atom_index, to_atom_index = bond.atom_indices
        from_subgraph_atom_index = atom_map[from_atom_index]
        to_subgraph_atom_index = atom_map[to_atom_index]
        from_node = cangen_nodes[from_subgraph_atom_index]
        to_node = cangen_nodes[to_subgraph_atom_index]
        from_node.neighbors.append(to_node)
        to_node.neighbors.append(from_node)
        canonical_bondtype = bond.canonical_bondtype
        canonical_labels[from_subgraph_atom_index].append(canonical_bondtype)
        canonical_labels[to_subgraph_atom_index].append(canonical_bondtype)
        from_node.outgoing_edges.append(OutgoingEdge(from_subgraph_atom_index, bond_index, bond.bond_smarts, to_subgraph_atom_index, to_node))
        to_node.outgoing_edges.append(OutgoingEdge(to_subgraph_atom_index, bond_index, bond.bond_smarts, from_subgraph_atom_index, from_node))
    if do_initial_assignment:
        for atom_index, node, canonical_label in zip(subgraph.atom_indices, cangen_nodes, canonical_labels):
            canonical_label.sort()
            canonical_label.append(atoms[atom_index].atom_smarts)
            label = '\n'.join(canonical_label)
            node.value = atom_assignment[label]
    return cangen_nodes