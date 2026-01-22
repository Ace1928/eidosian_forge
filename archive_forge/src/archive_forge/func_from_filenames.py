from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
@classmethod
def from_filenames(cls, poscar_filenames, transformations=None, extend_collection=False) -> StandardTransmuter:
    """Convenient constructor to generates a POSCAR transmuter from a list of
        POSCAR filenames.

        Args:
            poscar_filenames: List of POSCAR filenames
            transformations: New transformations to be applied to all
                structures.
            extend_collection:
                Same meaning as in __init__.
        """
    trafo_structs = []
    for filename in poscar_filenames:
        with open(filename, encoding='utf-8') as file:
            trafo_structs.append(TransformedStructure.from_poscar_str(file.read(), []))
    return StandardTransmuter(trafo_structs, transformations, extend_collection=extend_collection)