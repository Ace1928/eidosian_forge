from __future__ import annotations
import collections
import itertools
import json
import logging
import os
import platform
import sys
import warnings
from typing import TYPE_CHECKING
import requests
from monty.json import MontyDecoder
from pymatgen.core import SETTINGS
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_structure_by_material_id(self, material_id: str, conventional_unit_cell: bool=False) -> Structure:
    """
        Get a Structure corresponding to a material_id.

        Args:
            material_id (str): Materials Project ID (e.g. mp-1234).
            final (bool): Whether to get the final structure, or the initial
                (pre-relaxation) structures. Defaults to True.
            conventional_unit_cell (bool): Whether to get the standard conventional unit cell

        Returns:
            Structure object.
        """
    prop = 'structure'
    response = self.request(f'materials/summary/{material_id}/?_fields={prop}')
    structure = response[0][prop]
    if conventional_unit_cell:
        return SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    return structure