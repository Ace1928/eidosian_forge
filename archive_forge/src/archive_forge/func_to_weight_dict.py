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
@property
def to_weight_dict(self) -> dict[str, float]:
    """
        Returns:
            dict[str, float] with weight fraction of each component {"Ti": 0.90, "V": 0.06, "Al": 0.04}.
        """
    return {str(el): self.get_wt_fraction(el) for el in self.elements}