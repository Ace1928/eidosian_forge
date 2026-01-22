import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def get_bond_ring_score(bond_data, atoms=mol.atoms):
    bond_index, bond = bond_data
    a1, a2 = bond.atom_indices
    return bond.is_in_ring + atoms[a1].is_in_ring + atoms[a2].is_in_ring