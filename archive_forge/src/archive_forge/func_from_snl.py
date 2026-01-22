from __future__ import annotations
import datetime
import json
import re
from typing import TYPE_CHECKING, Any
from warnings import warn
from monty.json import MSONable, jsanitize
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.util.provenance import StructureNL
@classmethod
def from_snl(cls, snl: StructureNL) -> Self:
    """Create TransformedStructure from SNL.

        Args:
            snl (StructureNL): Starting snl

        Returns:
            TransformedStructure
        """
    history: list[dict] = []
    for hist in snl.history:
        dct = hist.description
        dct['_snl'] = {'url': hist.url, 'name': hist.name}
        history.append(dct)
    return cls(snl.structure, history=history)