from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
@property
def number_of_permutations(self):
    """Returns the number of permutations of this coordination geometry."""
    if self.permutations_safe_override:
        return factorial(self.coordination)
    if self.permutations is None:
        return factorial(self.coordination)
    return len(self.permutations)