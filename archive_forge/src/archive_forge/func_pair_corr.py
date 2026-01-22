from __future__ import annotations
import functools
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from operator import mul
from typing import TYPE_CHECKING
from monty.design_patterns import cached_class
from pymatgen.core import Species, get_el_sp
from pymatgen.util.due import Doi, due
def pair_corr(self, s1, s2):
    """
        Pair correlation of two species.

        Returns:
            The pair correlation of 2 species
        """
    return math.exp(self.get_lambda(s1, s2)) * self.Z / (self.get_px(s1) * self.get_px(s2))