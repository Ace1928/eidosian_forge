from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def parse_scalar(self, property: str) -> float | None:
    """Parse a scalar property from the chunk.

        Args:
            property (str): The property key to parse

        Returns:
            The scalar value of the property or None if not found
        """
    line_start = self.reverse_search_for(SCALAR_PROPERTY_TO_LINE_KEY[property])
    if line_start == LINE_NOT_FOUND:
        return None
    line = self.lines[line_start]
    return float(line.split(':')[-1].strip().split()[0])