from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
@property
def working_ion(self):
    """Working ion as pymatgen Element object."""
    return self.working_ion_entry.elements[0]