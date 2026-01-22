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
@staticmethod
def make_full_bse_densities_folder(folder):
    """Mkdir "FULL_BSE_Densities" folder (needed for bse run) in the desired folder."""
    if os.path.isfile(f'{folder}/FULL_BSE_Densities'):
        return 'FULL_BSE_Densities folder already exists'
    os.makedirs(f'{folder}/FULL_BSE_Densities')
    return 'makedirs FULL_BSE_Densities folder'