from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
def extend_transformations(self, transformations):
    """Extends a sequence of transformations to the TransformedStructure.

        Args:
            transformations: Sequence of Transformations
        """
    for trafo in transformations:
        self.append_transformation(trafo)