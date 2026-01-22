import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def remove_unknown_bondtypes(typed_mol, supported_canonical_bondtypes):
    emol = Chem.EditableMol(Chem.Mol())
    for atom in typed_mol.rdmol_atoms:
        emol.AddAtom(atom)
    orig_bonds = []
    new_bond_smarts_types = []
    new_canonical_bondtypes = []
    for bond, bond_smarts, canonical_bondtype in zip(typed_mol.rdmol_bonds, typed_mol.bond_smarts_types, typed_mol.canonical_bondtypes):
        if canonical_bondtype in supported_canonical_bondtypes:
            orig_bonds.append(bond)
            new_bond_smarts_types.append(bond_smarts)
            new_canonical_bondtypes.append(canonical_bondtype)
            emol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    new_mol = emol.GetMol()
    return FragmentedTypedMolecule(new_mol, list(new_mol.GetAtoms()), typed_mol.rdmol_atoms, orig_bonds, typed_mol.atom_smarts_types, new_bond_smarts_types, new_canonical_bondtypes)