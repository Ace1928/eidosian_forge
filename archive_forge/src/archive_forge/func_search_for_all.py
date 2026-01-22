from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def search_for_all(self, key: str, line_start: int=0, line_end: int=-1) -> list[int]:
    """Find the all times the key appears in self.lines.

        Args:
            key (str): The key string to search for in self.lines
            line_start (int): The first line to start the search from
            line_end (int): The last line to end the search at

        Returns:
            All times the key appears in the lines
        """
    line_index = []
    for ll, line in enumerate(self.lines[line_start:line_end]):
        if key in line:
            line_index.append(ll + line_start)
    return line_index