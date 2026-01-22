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
def from_cif_str(cls, cif_string: str, transformations: list[AbstractTransformation] | None=None, primitive: bool=True, occupancy_tolerance: float=1.0) -> Self:
    """Generates TransformedStructure from a cif string.

        Args:
            cif_string (str): Input cif string. Should contain only one
                structure. For CIFs containing multiple structures, please use
                CifTransmuter.
            transformations (list[Transformation]): Sequence of transformations
                to be applied to the input structure.
            primitive (bool): Option to set if the primitive cell should be
                extracted. Defaults to True. However, there are certain
                instances where you might want to use a non-primitive cell,
                e.g., if you are trying to generate all possible orderings of
                partial removals or order a disordered structure. Defaults to True.
            occupancy_tolerance (float): If total occupancy of a site is
                between 1 and occupancy_tolerance, the occupancies will be
                scaled down to 1.

        Returns:
            TransformedStructure
        """
    parser = CifParser.from_str(cif_string, occupancy_tolerance=occupancy_tolerance)
    raw_string = re.sub("'", '"', cif_string)
    cif_dict = parser.as_dict()
    cif_keys = list(cif_dict)
    struct = parser.parse_structures(primitive=primitive)[0]
    partial_cif = cif_dict[cif_keys[0]]
    if '_database_code_ICSD' in partial_cif:
        source = partial_cif['_database_code_ICSD'] + '-ICSD'
    else:
        source = 'uploaded cif'
    source_info = {'source': source, 'datetime': str(datetime.datetime.now()), 'original_file': raw_string, 'cif_data': cif_dict[cif_keys[0]]}
    return cls(struct, transformations, history=[source_info])