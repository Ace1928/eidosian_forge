from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.composition import Composition
@property
def is_element(self) -> bool:
    """Whether composition of entry is an element."""
    return self._composition.is_element