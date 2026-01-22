from __future__ import annotations
import collections
import os
import re
import string
import warnings
from functools import total_ordering
from itertools import combinations_with_replacement, product
from math import isnan
from typing import TYPE_CHECKING, cast
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.periodic_table import DummySpecies, Element, ElementType, Species, get_el_sp
from pymatgen.core.units import Mass
from pymatgen.util.string import Stringify, formula_double_format
def get_sym_dict(form: str, factor: float) -> dict[str, float]:
    sym_dict: dict[str, float] = collections.defaultdict(float)
    for m in re.finditer('([A-Z][a-z]*)\\s*([-*\\.e\\d]*)', form):
        el = m.group(1)
        amt = 1.0
        if m.group(2).strip() != '':
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        form = form.replace(m.group(), '', 1)
    if form.strip():
        raise ValueError(f'{form} is an invalid formula!')
    return sym_dict