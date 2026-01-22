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
def to_data_dict(self) -> dict:
    """
        Returns:
            A dict with many keys and values relating to Composition/Formula,
            including reduced_cell_composition, unit_cell_composition,
            reduced_cell_formula, elements and nelements.
        """
    return {'reduced_cell_composition': self.reduced_composition, 'unit_cell_composition': self.as_dict(), 'reduced_cell_formula': self.reduced_formula, 'elements': list(map(str, self)), 'nelements': len(self)}