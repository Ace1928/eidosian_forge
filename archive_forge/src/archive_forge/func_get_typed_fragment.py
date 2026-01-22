import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def get_typed_fragment(typed_mol, atom_indices):
    rdmol = typed_mol.rdmol
    rdmol_atoms = typed_mol.rdmol_atoms
    emol = Chem.EditableMol(Chem.Mol())
    atom_smarts_types = []
    atom_map = {}
    for i, atom_index in enumerate(atom_indices):
        atom = rdmol_atoms[atom_index]
        emol.AddAtom(atom)
        atom_smarts_types.append(typed_mol.atom_smarts_types[atom_index])
        atom_map[atom_index] = i
    orig_bonds = []
    bond_smarts_types = []
    new_canonical_bondtypes = []
    for bond, orig_bond, bond_smarts, canonical_bondtype in zip(rdmol.GetBonds(), typed_mol.orig_bonds, typed_mol.bond_smarts_types, typed_mol.canonical_bondtypes):
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        count = (begin_atom_idx in atom_map) + (end_atom_idx in atom_map)
        if count == 2:
            bond_smarts_types.append(bond_smarts)
            new_canonical_bondtypes.append(canonical_bondtype)
            emol.AddBond(atom_map[begin_atom_idx], atom_map[end_atom_idx], bond.GetBondType())
            orig_bonds.append(orig_bond)
        elif count == 1:
            raise AssertionError('connected/disconnected atoms?')
    return TypedFragment(emol.GetMol(), [typed_mol.orig_atoms[atom_index] for atom_index in atom_indices], orig_bonds, atom_smarts_types, bond_smarts_types, new_canonical_bondtypes)