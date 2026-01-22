from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def reverse_search_for(self, keys: list[str], line_start: int=0) -> int:
    """Find the last time one of the keys appears in self.lines.

        Args:
            keys (list[str]): The key strings to search for in self.lines
            line_start (int): The lowest index to search for in self.lines

        Returns:
            The last time one of the keys appears in self.lines
        """
    for idx, line in enumerate(self.lines[line_start:][::-1]):
        if any((key in line for key in keys)):
            return len(self.lines) - idx - 1
    return LINE_NOT_FOUND