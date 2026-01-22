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
def print_dos(self, ndigits=6) -> None:
    """
        Activate printing of the overall DOS file.

        Note: As of 2022.1, ndigits needs to be set to a sufficient value to ensure data is not lost.
        Note: As of 2022.1, can only be used with a k-point calculation.
        """
    if self.kpoints:
        self['force_eval']['dft']['print'].insert(DOS(ndigits))