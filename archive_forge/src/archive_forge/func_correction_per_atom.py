from __future__ import annotations
import abc
import json
import math
import os
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from scipy.interpolate import interp1d
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.due import Doi, due
@property
def correction_per_atom(self) -> float:
    """
        Returns:
            float: the total energy correction / adjustment applied to the entry in eV/atom.
        """
    return self.correction / self.composition.num_atoms