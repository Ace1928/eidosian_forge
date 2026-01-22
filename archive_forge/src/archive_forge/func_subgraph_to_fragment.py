import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def subgraph_to_fragment(mol, subgraph):
    emol = Chem.EditableMol(Chem.Mol())
    atom_map = {}
    for atom_index in subgraph.atom_indices:
        emol.AddAtom(mol.GetAtomWithIdx(atom_index))
        atom_map[atom_index] = len(atom_map)
    for bond_index in subgraph.bond_indices:
        bond = mol.GetBondWithIdx(bond_index)
        emol.AddBond(atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()], bond.GetBondType())
    return emol.GetMol()