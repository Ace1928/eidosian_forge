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
@property
def types_of_magnetic_species(self) -> tuple[Element | Species | DummySpecies, ...]:
    """Equivalent to Structure.types_of_specie but only returns magnetic species.

        Returns:
            tuple: types of Species
        """
    if self.number_of_magnetic_sites > 0:
        structure = self.get_structure_with_only_magnetic_atoms()
        return tuple(sorted(structure.types_of_species))
    return ()