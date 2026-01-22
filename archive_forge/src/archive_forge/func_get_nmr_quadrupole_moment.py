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
def get_nmr_quadrupole_moment(self, isotope: str | None=None) -> float:
    """Gets the nuclear electric quadrupole moment in units of e * millibarns.

        Args:
            isotope (str): the isotope to get the quadrupole moment for
                default is None, which gets the lowest mass isotope
        """
    quad_mom = self._el.nmr_quadrupole_moment
    if not quad_mom:
        return 0.0
    if isotope is None:
        isotopes = list(quad_mom)
        isotopes.sort(key=lambda x: int(x.split('-')[1]), reverse=False)
        return quad_mom.get(isotopes[0], 0.0)
    if isotope not in quad_mom:
        raise ValueError(f'No quadrupole moment for isotope={isotope!r}')
    return quad_mom.get(isotope, 0.0)