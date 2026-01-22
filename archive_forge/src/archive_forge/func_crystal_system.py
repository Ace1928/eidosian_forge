from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, overload
import numpy as np
from monty.design_patterns import cached_class
from monty.serialization import loadfn
from pymatgen.util.string import Stringify
@property
def crystal_system(self) -> CrystalSystem:
    """
        Returns:
            str: Crystal system of the space group, e.g., cubic, hexagonal, etc.
        """
    num = self.int_number
    if num <= 2:
        return 'triclinic'
    if num <= 15:
        return 'monoclinic'
    if num <= 74:
        return 'orthorhombic'
    if num <= 142:
        return 'tetragonal'
    if num <= 167:
        return 'trigonal'
    if num <= 194:
        return 'hexagonal'
    return 'cubic'