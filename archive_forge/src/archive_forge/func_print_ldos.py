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
def print_ldos(self, nlumo: int=-1) -> None:
    """
        Activate the printing of LDOS files, printing one for each atom kind by default.

        Args:
            nlumo (int): Number of virtual orbitals to be added to the MO set (-1=all).
                CAUTION: Setting this value to be higher than the number of states present may
                cause a Cholesky error.
        """
    if not self.check('FORCE_EVAL/DFT/PRINT/PDOS'):
        self['FORCE_EVAL']['DFT']['PRINT'].insert(PDOS(nlumo=nlumo))
    for idx in range(len(self.structure)):
        self['FORCE_EVAL']['DFT']['PRINT']['PDOS'].insert(LDOS(idx + 1, alias=f'LDOS {idx + 1}', verbose=False))