from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@classmethod
def from_voigt(cls, voigt_input_list, base_class=Tensor) -> Self:
    """Creates TensorCollection from voigt form.

        Args:
            voigt_input_list: List of voigt tensors
            base_class: Class for tensor.

        Returns:
            TensorCollection.
        """
    return cls([base_class.from_voigt(v) for v in voigt_input_list])