from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.io.aims.parsers import (
@property
def structures(self) -> Sequence[Structure | Molecule]:
    """All images in the output file."""
    return self._results