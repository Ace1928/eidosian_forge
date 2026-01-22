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
@classmethod
def from_weight_dict(cls, weight_dict: dict[SpeciesLike, float]) -> Self:
    """Creates a Composition based on a dict of atomic fractions calculated
        from a dict of weight fractions. Allows for quick creation of the class
        from weight-based notations commonly used in the industry, such as
        Ti6V4Al and Ni60Ti40.

        Args:
            weight_dict (dict): {symbol: weight_fraction} dict.

        Returns:
            Composition
        """
    weight_sum = sum((val / Element(el).atomic_mass for el, val in weight_dict.items()))
    comp_dict = {el: val / Element(el).atomic_mass / weight_sum for el, val in weight_dict.items()}
    return cls(comp_dict)