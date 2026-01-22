from __future__ import annotations
import abc
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_spg_num(struct: Structure) -> int:
    finder = SpacegroupAnalyzer(struct, symprec=self.symprec)
    return finder.get_space_group_number()