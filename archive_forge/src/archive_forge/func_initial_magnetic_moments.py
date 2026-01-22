from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def initial_magnetic_moments(self) -> Sequence[float]:
    """The initial magnetic Moments"""
    if 'initial_magnetic_moments' not in self._cache:
        self._parse_initial_charges_and_moments()
    return self._cache['initial_magnetic_moments']