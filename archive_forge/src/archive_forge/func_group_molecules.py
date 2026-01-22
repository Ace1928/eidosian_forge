from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
def group_molecules(self, mol_list):
    """
        Group molecules by structural equality.

        Args:
            mol_list: List of OpenBabel OBMol or pymatgen objects

        Returns:
            A list of lists of matched molecules
            Assumption: if s1=s2 and s2=s3, then s1=s3
            This may not be true for small tolerances.
        """
    mol_hash = [(idx, self._mapper.get_molecule_hash(mol)) for idx, mol in enumerate(mol_list)]
    mol_hash.sort(key=lambda x: x[1])
    raw_groups = tuple((tuple((m[0] for m in g)) for k, g in itertools.groupby(mol_hash, key=lambda x: x[1])))
    group_indices = []
    for rg in raw_groups:
        mol_eq_test = [(p[0], p[1], self.fit(mol_list[p[0]], mol_list[p[1]])) for p in itertools.combinations(sorted(rg), 2)]
        mol_eq = {(p[0], p[1]) for p in mol_eq_test if p[2]}
        not_alone_mols = set(itertools.chain.from_iterable(mol_eq))
        alone_mols = set(rg) - not_alone_mols
        group_indices.extend([[m] for m in alone_mols])
        while len(not_alone_mols) > 0:
            current_group = {not_alone_mols.pop()}
            while len(not_alone_mols) > 0:
                candidate_pairs = {tuple(sorted(p)) for p in itertools.product(current_group, not_alone_mols)}
                mutual_pairs = candidate_pairs & mol_eq
                if len(mutual_pairs) == 0:
                    break
                mutual_mols = set(itertools.chain.from_iterable(mutual_pairs))
                current_group |= mutual_mols
                not_alone_mols -= mutual_mols
            group_indices.append(sorted(current_group))
    group_indices.sort(key=lambda x: (len(x), -x[0]), reverse=True)
    return [[mol_list[idx] for idx in g] for g in group_indices]