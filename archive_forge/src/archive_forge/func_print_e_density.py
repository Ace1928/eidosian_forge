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
def print_e_density(self, stride=(2, 2, 2)) -> None:
    """Controls the printing of cube files with electronic density and, for UKS, the spin density."""
    if not self.check('FORCE_EVAL/DFT/PRINT/E_DENSITY_CUBE'):
        self['FORCE_EVAL']['DFT']['PRINT'].insert(EDensityCube(keywords={'STRIDE': Keyword('STRIDE', *stride)}))