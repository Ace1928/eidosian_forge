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
def get_px(self, sp):
    """
        Args:
            sp (Species/Element): Species.

        Returns:
            Probability
        """
    return self._px[get_el_sp(sp)]