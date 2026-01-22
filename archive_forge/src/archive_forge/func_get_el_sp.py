from __future__ import annotations
import ast
import functools
import json
import re
import warnings
from collections import Counter
from enum import Enum, unique
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.units import SUPPORTED_UNIT_NAMES, FloatWithUnit, Ha_to_eV, Length, Mass, Unit
from pymatgen.util.string import Stringify, formula_double_format
@functools.lru_cache
def get_el_sp(obj: int | SpeciesLike) -> Element | Species | DummySpecies:
    """Utility method to get an Element, Species or DummySpecies from any input.

    If obj is in itself an element or a specie, it is returned automatically.
    If obj is an int or a string representing an integer, the Element with the
    atomic number obj is returned.
    If obj is a string, Species parsing will be attempted (e.g. Mn2+). Failing that
    Element parsing will be attempted (e.g. Mn). Failing that DummyElement parsing
    will be attempted.

    Args:
        obj (Element/Species/str/int): An arbitrary object. Supported objects
            are actual Element/Species objects, integers (representing atomic
            numbers) or strings (element symbols or species strings).

    Raises:
        ValueError: if obj cannot be converted into an Element or Species.

    Returns:
        Species | Element: with a bias for the maximum number of properties
            that can be determined.
    """
    if isinstance(obj, (Element, Species, DummySpecies)):
        if getattr(obj, '_is_named_isotope', None):
            return Element(obj.name) if isinstance(obj, Element) else Species(str(obj))
        return obj
    try:
        flt = float(obj)
        assert flt == int(flt)
        return Element.from_Z(int(flt))
    except (AssertionError, ValueError, TypeError, KeyError):
        pass
    try:
        return Species.from_str(obj)
    except (ValueError, TypeError, KeyError):
        pass
    try:
        return Element(obj)
    except (ValueError, TypeError, KeyError):
        pass
    try:
        return DummySpecies.from_str(obj)
    except Exception:
        raise ValueError(f"Can't parse Element or Species from {obj!r}")