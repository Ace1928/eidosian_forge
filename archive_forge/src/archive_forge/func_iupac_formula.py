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
def iupac_formula(self) -> str:
    """Returns a formula string, with elements sorted by the iupac
        electronegativity ordering defined in Table VI of "Nomenclature of
        Inorganic Chemistry (IUPAC Recommendations 2005)". This ordering
        effectively follows the groups and rows of the periodic table, except
        the Lanthanides, Actinides and hydrogen. Polyanions are still determined
        based on the true electronegativity of the elements.
        e.g. CH2(SO4)2.
        """
    sym_amt = self.get_el_amt_dict()
    syms = sorted(sym_amt, key=lambda s: get_el_sp(s).iupac_ordering)
    formula = [f'{s}{formula_double_format(sym_amt[s], ignore_ones=False)}' for s in syms]
    return ' '.join(formula)