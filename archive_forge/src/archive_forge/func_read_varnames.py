from __future__ import annotations
import logging
import os.path
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.dev import requires
from monty.functools import lazy_property
from monty.string import marquee
from pymatgen.core.structure import Structure
from pymatgen.core.units import ArrayWithUnit
from pymatgen.core.xcfunc import XcFunc
def read_varnames(self, path='/'):
    """List of variable names stored in the group specified by path."""
    if path == '/':
        return list(self.rootgrp.variables)
    group = self.path2group[path]
    return list(group.variables)