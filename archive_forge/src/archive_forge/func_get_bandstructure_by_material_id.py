from __future__ import annotations
import itertools
import json
import logging
import math
import os
import platform
import re
import sys
import warnings
from enum import Enum, unique
from time import sleep
from typing import TYPE_CHECKING, Any, Literal
import requests
from monty.json import MontyDecoder, MontyEncoder
from ruamel.yaml import YAML
from tqdm import tqdm
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.exp_entries import ExpEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def get_bandstructure_by_material_id(self, material_id, line_mode=True):
    """Get a BandStructure corresponding to a material_id.

        REST Endpoint: https://materialsproject.org/rest/v2/materials/<mp-id>/vasp/bandstructure or
        https://materialsproject.org/rest/v2/materials/<mp-id>/vasp/bandstructure_uniform

        Args:
            material_id (str): Materials Project material_id.
            line_mode (bool): If True, fetch a BandStructureSymmLine object
                (default). If False, return the uniform band structure.

        Returns:
            A BandStructure object.
        """
    prop = 'bandstructure' if line_mode else 'bandstructure_uniform'
    data = self.get_data(material_id, prop=prop)
    return data[0][prop]