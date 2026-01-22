from __future__ import annotations
import itertools
import os
import re
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any
import numpy as np
import spglib
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
def get_all_possible_basis_combinations(min_basis: list, max_basis: list) -> list:
    """

    Args:
        min_basis: list of basis entries: e.g., ['Si 3p 3s ']
        max_basis: list of basis entries: e.g., ['Si 3p 3s '].

    Returns:
        list[list[str]]: all possible combinations of basis functions, e.g. [['Si 3p 3s']]
    """
    max_basis_lists = [x.split() for x in max_basis]
    min_basis_lists = [x.split() for x in min_basis]
    basis_dict: dict[str, dict] = {}
    for iel, el in enumerate(max_basis_lists):
        basis_dict[el[0]] = {'fixed': [], 'variable': [], 'combinations': []}
        for basis in el[1:]:
            if basis in min_basis_lists[iel]:
                basis_dict[el[0]]['fixed'].append(basis)
            if basis not in min_basis_lists[iel]:
                basis_dict[el[0]]['variable'].append(basis)
        for L in range(len(basis_dict[el[0]]['variable']) + 1):
            for subset in itertools.combinations(basis_dict[el[0]]['variable'], L):
                basis_dict[el[0]]['combinations'].append(' '.join([el[0]] + basis_dict[el[0]]['fixed'] + list(subset)))
    list_basis = [item['combinations'] for item in basis_dict.values()]
    start_basis = list_basis[0]
    if len(list_basis) > 1:
        for el in list_basis[1:]:
            new_start_basis = []
            for elbasis in start_basis:
                for elbasis2 in el:
                    if not isinstance(elbasis, list):
                        new_start_basis.append([elbasis, elbasis2])
                    else:
                        new_start_basis.append([*elbasis.copy(), elbasis2])
            start_basis = new_start_basis
        return start_basis
    return [[basis] for basis in start_basis]