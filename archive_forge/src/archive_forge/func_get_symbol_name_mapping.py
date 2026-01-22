from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def get_symbol_name_mapping(self, coordination=None):
    """
        Return a dictionary mapping the symbol of a CoordinationGeometry to its name.

        Args:
            coordination: Whether to restrict the dictionary to a given coordination.

        Returns:
            dict: map symbol of a CoordinationGeometry to its name.
        """
    geom = {}
    if coordination is None:
        for gg in self.cg_list:
            geom[gg.mp_symbol] = gg.name
    else:
        for gg in self.cg_list:
            if gg.get_coordination_number() == coordination:
                geom[gg.mp_symbol] = gg.name
    return geom