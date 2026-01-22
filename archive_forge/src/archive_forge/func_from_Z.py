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
@staticmethod
def from_Z(Z: int, A: int | None=None) -> Element:
    """Get an element from an atomic number.

        Args:
            Z (int): Atomic number (number of protons)
            A (int | None) : Atomic mass number (number of protons + neutrons)

        Returns:
            Element with atomic number Z.
        """
    for sym, data in _pt_data.items():
        atomic_mass_num = data.get('Atomic mass no') if A else None
        if data['Atomic no'] == Z and atomic_mass_num == A:
            return Element(sym)
    raise ValueError(f'Unexpected atomic number Z={Z!r}')