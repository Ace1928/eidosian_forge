from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def linked_against(self) -> list[str]:
    """Get all libraries used to link the FHI-aims executable."""
    line_start = self.reverse_search_for(['Linking against:'])
    if line_start == LINE_NOT_FOUND:
        return []
    linked_libs = [self.lines[line_start].split(':')[1].strip()]
    line_start += 1
    while 'lib' in self.lines[line_start]:
        linked_libs.append(self.lines[line_start].strip())
        line_start += 1
    return linked_libs