from __future__ import annotations
import copy
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.inputs import AimsControlIn, AimsGeometryIn
from pymatgen.io.aims.parsers import AimsParseError, read_aims_output
from pymatgen.io.core import InputFile, InputGenerator, InputSet
def get_input_set(self, structure: Structure | Molecule | None=None, prev_dir: str | Path | None=None, properties: list[str] | None=None) -> AimsInputSet:
    """Generate an AimsInputSet object.

        Parameters
        ----------
        structure : Structure or Molecule
            Structure or Molecule to generate the input set for.
        prev_dir: str or Path
            Path to the previous working directory
        properties: list[str]
            System properties that are being calculated

        Returns:
            AimsInputSet: The input set for the calculation of structure
        """
    prev_structure, prev_parameters, _ = self._read_previous(prev_dir)
    structure = structure or prev_structure
    if structure is None:
        raise ValueError('No structure can be determined to generate the input set')
    parameters = self._get_input_parameters(structure, prev_parameters)
    properties = self._get_properties(properties, parameters)
    return AimsInputSet(parameters=parameters, structure=structure, properties=properties)