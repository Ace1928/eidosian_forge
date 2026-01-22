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
def hill_formula(self) -> str:
    """The Hill system (or Hill notation) is a system of writing empirical chemical
        formulas, molecular chemical formulas and components of a condensed formula such
        that the number of carbon atoms in a molecule is indicated first, the number of
        hydrogen atoms next, and then the number of all other chemical elements
        subsequently, in alphabetical order of the chemical symbols. When the formula
        contains no carbon, all the elements, including hydrogen, are listed
        alphabetically.
        """
    elem_comp = self.element_composition
    elements = sorted((el.symbol for el in elem_comp))
    hill_elements = []
    if 'C' in elements:
        hill_elements.append('C')
        elements.remove('C')
        if 'H' in elements:
            hill_elements.append('H')
            elements.remove('H')
    hill_elements += elements
    formula = [f'{el}{(formula_double_format(elem_comp[el]) if elem_comp[el] != 1 else '')}' for el in hill_elements]
    return ' '.join(formula)