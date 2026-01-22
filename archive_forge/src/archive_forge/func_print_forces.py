from __future__ import annotations
import itertools
import os
import warnings
import numpy as np
from ruamel.yaml import YAML
from pymatgen.core import SETTINGS
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Element, Molecule, Structure
from pymatgen.io.cp2k.inputs import (
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff, get_unique_site_indices
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
def print_forces(self) -> None:
    """Print out the forces and stress during calculation."""
    self['FORCE_EVAL'].insert(Section('PRINT', subsections={}))
    self['FORCE_EVAL']['PRINT'].insert(Section('FORCES', subsections={}))
    self['FORCE_EVAL']['PRINT'].insert(Section('STRESS_TENSOR', subsections={}))