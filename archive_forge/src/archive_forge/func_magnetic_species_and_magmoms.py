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
def magnetic_species_and_magmoms(self) -> dict[str, Any]:
    """Returns a dict of magnetic species and the magnitude of
        their associated magmoms. Will return a list if there are
        multiple magmoms per species.

        Returns:
            dict of magnetic species and magmoms
        """
    structure = self.get_ferromagnetic_structure()
    mag_types: dict = {str(site.specie): set() for site in structure if site.properties['magmom'] != 0}
    for site in structure:
        if site.properties['magmom'] != 0:
            mag_types[str(site.specie)].add(site.properties['magmom'])
    for sp, magmoms in mag_types.items():
        if len(magmoms) == 1:
            mag_types[sp] = magmoms.pop()
        else:
            mag_types[sp] = sorted(magmoms)
    return mag_types