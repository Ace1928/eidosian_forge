from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def initial_charges(self) -> Sequence[float]:
    """The initial charges for the structure"""
    if 'initial_charges' not in self._cache:
        self._parse_initial_charges_and_moments()
    return self._cache['initial_charges']