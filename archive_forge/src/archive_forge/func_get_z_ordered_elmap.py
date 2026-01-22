from __future__ import annotations
import collections
import functools
import operator
import os
from math import exp, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_z_ordered_elmap(comp):
    """
    Arbitrary ordered element map on the elements/species of a composition of a
    given site in an unordered structure. Returns a list of tuples (
    element_or_specie: occupation) in the arbitrary order.

    The arbitrary order is based on the Z of the element and the smallest
    fractional occupations first.
    Example : {"Ni3+": 0.2, "Ni4+": 0.2, "Cr3+": 0.15, "Zn2+": 0.34,
    "Cr4+": 0.11} will yield the species in the following order :
    Cr4+, Cr3+, Ni3+, Ni4+, Zn2+ ... or
    Cr4+, Cr3+, Ni4+, Ni3+, Zn2+
    """
    return sorted(((elem, comp[elem]) for elem in comp))