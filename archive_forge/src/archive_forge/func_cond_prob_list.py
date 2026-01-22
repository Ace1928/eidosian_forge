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
def cond_prob_list(self, l1, l2):
    """
        Find the probabilities of 2 lists. These should include ALL species.
        This is the probability conditional on l2.

        Args:
            l1, l2:
                lists of species

        Returns:
            The conditional probability (assuming these species are in
            l2)
        """
    assert len(l1) == len(l2)
    p = 1
    for s1, s2 in zip(l1, l2):
        p *= self.cond_prob(s1, s2)
    return p