from __future__ import annotations
import os
from typing import TYPE_CHECKING
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.util.due import Doi, due
def get_prototypes(self, structure: Structure) -> list | None:
    """
        Get prototype(s) structures for a given input structure. If you use this method in
        your work, please cite the appropriate AFLOW publication:

        Mehl, M. J., Hicks, D., Toher, C., Levy, O., Hanson, R. M., Hart, G., & Curtarolo,
        S. (2017). The AFLOW library of crystallographic prototypes: part 1. Computational
        Materials Science, 136, S1-S828. https://doi.org/10.1016/j.commatsci.2017.01.017

        Args:
            structure: structure to match

        Returns:
            list | None: A list of dicts with keys 'snl' for the matched prototype and
                'tags', a dict of tags ('mineral', 'strukturbericht' and 'aflow') of that
                prototype. This should be a list containing just a single entry, but it is
                possible a material can match multiple prototypes.
        """
    tags = self._match_single_prototype(structure)
    return tags or None