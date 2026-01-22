from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.io.aims.parsers import (
@property
def final_structure(self) -> Structure | Molecule:
    """The final structure for the calculation."""
    return self._results[-1]