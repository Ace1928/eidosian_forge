import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def assign_isotopes_from_class_tag(mol, atom_class_tag):
    try:
        atom_classes = mol.GetProp(atom_class_tag)
    except KeyError:
        raise ValueError('Missing atom class tag %r' % (atom_class_tag,))
    fields = atom_classes.split()
    if len(fields) != mol.GetNumAtoms():
        raise ValueError('Mismatch between the number of atoms (#%d) and the number of atom classes (%d)' % (mol.GetNumAtoms(), len(fields)))
    new_isotopes = []
    for field in fields:
        if not field.isdigit():
            raise ValueError('Atom class %r from tag %r must be a number' % (field, atom_class_tag))
        isotope = int(field)
        if not 1 <= isotope <= 10000:
            raise ValueError('Atom class %r from tag %r must be in the range 1 to 10000' % (field, atom_class_tag))
        new_isotopes.append(isotope)
    save_isotopes(mol, get_isotopes(mol))
    save_atom_classes(mol, new_isotopes)
    set_isotopes(mol, new_isotopes)