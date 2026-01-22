from and back to a string/file is not guaranteed to be reversible, i.e. a diff on the output
from __future__ import annotations
import datetime
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.core import ParseError
@property
def spacegroup_label(self) -> str:
    """
        The Hermann-Mauguin notation of the spacegroup with ascii characters.
        So no. 225 would be Fm-3m, and no. 194 would be P6_3/mmc.
        """
    return self._TITL.spacegroup_label