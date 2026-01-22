from __future__ import annotations
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.serialization import loadfn
from ruamel.yaml.error import MarkedYAMLError
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from pymatgen.core.structure import DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation, MagOrderParameterConstraint
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.util.due import Doi, due
def get_nonmagnetic_structure(self, make_primitive: bool=True) -> Structure:
    """Returns a Structure without magnetic moments defined.

        Args:
            make_primitive: Whether to make structure primitive after
                removing magnetic information (Default value = True)

        Returns:
            Structure
        """
    structure = self.structure.copy()
    structure.remove_site_property('magmom')
    if make_primitive:
        structure = structure.get_primitive_structure()
    return structure