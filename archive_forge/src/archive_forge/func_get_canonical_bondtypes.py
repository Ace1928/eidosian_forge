import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def get_canonical_bondtypes(rdmol, bonds, atom_smarts_types, bond_smarts_types):
    canonical_bondtypes = []
    for bond, bond_smarts in zip(bonds, bond_smarts_types):
        atom1_smarts = atom_smarts_types[bond.GetBeginAtomIdx()]
        atom2_smarts = atom_smarts_types[bond.GetEndAtomIdx()]
        if atom1_smarts > atom2_smarts:
            atom1_smarts, atom2_smarts = (atom2_smarts, atom1_smarts)
        canonical_bondtypes.append('[%s]%s[%s]' % (atom1_smarts, bond_smarts, atom2_smarts))
    return canonical_bondtypes