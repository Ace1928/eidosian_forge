from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
@property
def x_discharge(self) -> float:
    """The number of working ions per formula unit of host in the discharged state."""
    return self.voltage_pairs[-1].x_discharge