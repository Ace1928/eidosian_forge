import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def set_isotopes(mol, isotopes):
    if mol.GetNumAtoms() != len(isotopes):
        raise ValueError('Mismatch between the number of atoms and the number of isotopes')
    for atom, isotope in zip(mol.GetAtoms(), isotopes):
        atom.SetMass(isotope)