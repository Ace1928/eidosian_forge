import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def get_selected_atom_classes(mol, atom_indices):
    atom_classes = _atom_class_dict.get(mol, None)
    if atom_classes is None:
        return None
    return [atom_classes[index] for index in atom_indices]