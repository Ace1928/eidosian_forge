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
def ionic_radius(self) -> float | None:
    """Ionic radius of specie. Returns None if data is not present."""
    if self._oxi_state in self.ionic_radii:
        return self.ionic_radii[self._oxi_state]
    if self._oxi_state:
        dct = self._el.data
        oxi_str = str(int(self._oxi_state))
        warn_msg = f'No default ionic radius for {self}.'
        if (ion_rad := dct.get('Ionic radii hs', {}).get(oxi_str)):
            warnings.warn(f'{warn_msg} Using hs data.')
            return ion_rad
        if (ion_rad := dct.get('Ionic radii ls', {}).get(oxi_str)):
            warnings.warn(f'{warn_msg} Using ls data.')
            return ion_rad
    warnings.warn(f'No ionic radius for {self}!')
    return None