from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def get_geometry_from_name(self, name):
    """
        Returns the coordination geometry of the given name.

        Args:
            name: The name of the coordination geometry.
        """
    for gg in self.cg_list:
        if gg.name == name or name in gg.alternative_names:
            return gg
    raise LookupError(f'No coordination geometry found with name {name!r}')