from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.tensors import SquareTensor

        Calculates the second Piola-Kirchoff stress.

        Args:
            def_grad (3x3 array-like): rate of deformation tensor
        