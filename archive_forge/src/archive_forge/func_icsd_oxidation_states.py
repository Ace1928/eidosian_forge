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
@property
def icsd_oxidation_states(self) -> tuple[int, ...]:
    """Tuple of all oxidation states with at least 10 instances in
        ICSD database AND at least 1% of entries for that element.
        """
    return tuple(self._data.get('ICSD oxidation states', []))