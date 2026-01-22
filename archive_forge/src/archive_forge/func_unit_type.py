from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
@property
def unit_type(self) -> str:
    """The type of unit. Energy, Charge, etc."""
    return self._unit_type