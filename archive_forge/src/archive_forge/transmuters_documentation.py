from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
Convenient constructor to generates a POSCAR transmuter from a list of
        POSCAR filenames.

        Args:
            poscar_filenames: List of POSCAR filenames
            transformations: New transformations to be applied to all
                structures.
            extend_collection:
                Same meaning as in __init__.
        