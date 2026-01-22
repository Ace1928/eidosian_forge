import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def find_upper_fragment_size_limits(rdmol, atoms):
    max_num_atoms = 0
    max_twice_num_bonds = 0
    for atom_indices in Chem.GetMolFrags(rdmol):
        max_num_atoms = max(max_num_atoms, len(atom_indices))
        twice_num_bonds = 0
        for atom_index in atom_indices:
            twice_num_bonds += len(atoms[atom_index].GetBonds())
        max_twice_num_bonds = max(max_twice_num_bonds, twice_num_bonds)
    return (max_num_atoms, max_twice_num_bonds // 2)