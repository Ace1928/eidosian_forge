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
def read_abinit_xcfunc(self):
    """Read ixc from an Abinit file. Return XcFunc object."""
    ixc = int(self.read_value('ixc'))
    return XcFunc.from_abinit_ixc(ixc)