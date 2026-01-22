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
def set_auxiliary_basis_set(self, folder, auxiliary_folder, auxiliary_basis_set_type='aug_cc_pvtz'):
    """
        copy in the desired folder the needed auxiliary basis set "X2.ion" where X is a specie.

        Args:
            auxiliary_folder: folder where the auxiliary basis sets are stored
            auxiliary_basis_set_type: type of basis set (string to be found in the extension of the file name; must
                be in lower case). ex: C2.ion_aug_cc_pvtz_RI_Weigend find "aug_cc_pvtz".
        """
    list_files = os.listdir(auxiliary_folder)
    for specie in self._mol.symbol_set:
        for file in list_files:
            if file.upper().find(specie.upper() + '2') != -1 and file.lower().find(auxiliary_basis_set_type) != -1:
                shutil.copyfile(f'{auxiliary_folder}/{file}', f'{folder}/{specie}2.ion')