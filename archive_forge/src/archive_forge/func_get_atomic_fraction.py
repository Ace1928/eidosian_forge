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
def get_atomic_fraction(self, el: SpeciesLike) -> float:
    """Calculate atomic fraction of an Element or Species.

        Args:
            el (Element/Species): Element or Species to get fraction for.

        Returns:
            Atomic fraction for element el in Composition
        """
    return abs(self[el]) / self._n_atoms