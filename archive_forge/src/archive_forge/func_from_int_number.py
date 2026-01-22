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
@classmethod
def from_int_number(cls, int_number: int, hexagonal: bool=True) -> Self:
    """Obtains a SpaceGroup from its international number.

        Args:
            int_number (int): International number.
            hexagonal (bool): For rhombohedral groups, whether to return the
                hexagonal setting (default) or rhombohedral setting.

        Raises:
            ValueError: If the international number is not valid, i.e. not between 1 and 230 inclusive.

        Returns:
            SpaceGroup: object with the given international number.
        """
    if int_number not in range(1, 231):
        raise ValueError(f'International number must be between 1 and 230, got {int_number}')
    symbol = sg_symbol_from_int_number(int_number, hexagonal=hexagonal)
    if not hexagonal and int_number in (146, 148, 155, 160, 161, 166, 167):
        symbol += ':R'
    return cls(symbol)