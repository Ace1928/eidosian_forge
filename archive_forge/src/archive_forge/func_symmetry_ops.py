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
def symmetry_ops(self) -> set[SymmOp]:
    """Full set of symmetry operations as matrices. Lazily initialized as
        generation sometimes takes a bit of time.
        """
    from pymatgen.core.operations import SymmOp
    if self._symmetry_ops is None:
        self._symmetry_ops = {SymmOp(m) for m in self._generate_full_symmetry_ops()}
    return self._symmetry_ops