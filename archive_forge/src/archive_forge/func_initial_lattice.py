from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def initial_lattice(self) -> Lattice | None:
    """The initial Lattice of the structure"""
    return self._header['initial_lattice']