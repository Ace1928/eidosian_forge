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
def cond_prob(self, s1, s2):
    """
        Conditional probability of substituting s1 for s2.

        Args:
            s1:
                The *variable* specie
            s2:
                The *fixed* specie

        Returns:
            Conditional probability used by structure predictor.
        """
    return math.exp(self.get_lambda(s1, s2)) / self.get_px(s2)