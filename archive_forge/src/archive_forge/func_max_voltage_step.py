from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
@property
def max_voltage_step(self):
    """Maximum absolute difference in adjacent voltage steps."""
    steps = [self.voltage_pairs[i].voltage - self.voltage_pairs[i + 1].voltage for i in range(len(self.voltage_pairs) - 1)]
    return max(steps) if len(steps) > 0 else 0