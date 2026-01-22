from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
@staticmethod
def get_default_units():
    """Returns: Default units."""
    return AdfKey.from_str('Units\nlength angstrom\nangle degree\nEnd')