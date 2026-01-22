import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def check_completeRingsOnly(smarts, subgraph, enumeration_mol):
    atoms = enumeration_mol.atoms
    bonds = enumeration_mol.bonds
    ring_bonds = []
    for bond_index in subgraph.bond_indices:
        bond = bonds[bond_index]
        if bond.is_in_ring:
            ring_bonds.append(bond_index)
    if not ring_bonds:
        return True
    if len(ring_bonds) <= 2:
        return False
    confirmed_ring_bonds = set()
    subgraph_ring_bond_indices = set(ring_bonds)
    for bond_index in ring_bonds:
        if bond_index in confirmed_ring_bonds:
            continue
        from_atom_index, to_atom_index = bonds[bond_index].atom_indices
        atom_depth = {from_atom_index: 0, to_atom_index: 1}
        bond_stack = [bond_index]
        backtrack_stack = []
        prev_bond_index = bond_index
        current_atom_index = to_atom_index
        while 1:
            next_bond_index = next_atom_index = None
            this_is_a_ring = False
            for outgoing_edge in enumeration_mol.directed_edges[current_atom_index]:
                if outgoing_edge.bond_index == prev_bond_index:
                    continue
                if outgoing_edge.bond_index not in subgraph_ring_bond_indices:
                    continue
                if outgoing_edge.end_atom_index in atom_depth:
                    confirmed_ring_bonds.update(bond_stack[atom_depth[outgoing_edge.end_atom_index]:])
                    confirmed_ring_bonds.add(outgoing_edge.bond_index)
                    if len(confirmed_ring_bonds) == len(ring_bonds):
                        return True
                    this_is_a_ring = True
                    continue
                if next_bond_index is None:
                    next_bond_index = outgoing_edge.bond_index
                    next_atom_index = outgoing_edge.end_atom_index
                else:
                    backtrack_stack.append((len(bond_stack), outgoing_edge.bond_index, outgoing_edge.end_atom_index))
            if next_bond_index is None:
                if this_is_a_ring:
                    while backtrack_stack:
                        old_size, prev_bond_index, current_atom_index = backtrack_stack.pop()
                        if bond_index not in confirmed_ring_bonds:
                            del bond_stack[old_size:]
                            break
                    else:
                        break
                else:
                    return False
            else:
                bond_stack.append(next_bond_index)
                atom_depth[next_atom_index] = len(bond_stack)
                prev_bond_index = next_bond_index
                current_atom_index = next_atom_index