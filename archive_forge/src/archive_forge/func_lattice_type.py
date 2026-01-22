from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.re import regrep
from pymatgen.core import Element, Lattice, Structure
from pymatgen.util.io_utils import clean_lines
@property
def lattice_type(self):
    """Returns: Lattice type."""
    return self.data['lattice_type']