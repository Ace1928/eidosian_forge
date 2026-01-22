from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
def set_n_nlmo(self):
    """the number of nlm orbitals for the basis set"""
    n_nlm_orbs = 0
    data_tmp = self.data
    data_tmp.pop('lmax')
    data_tmp.pop('n_nlo')
    data_tmp.pop('preamble')
    for l_zeta_ng in data_tmp:
        n_l = l_zeta_ng.split('_')[0]
        n_nlm_orbs = n_nlm_orbs + (2 * int(n_l) + 1)
    return str(n_nlm_orbs)