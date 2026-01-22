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
def parse_orbital(orb_str):
    m = re.match('(\\d+)([spdfg]+)(\\d+)', orb_str)
    if m:
        return (int(m.group(1)), m.group(2), int(m.group(3)))
    return orb_str