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
def dump_tddft_data_in_gw_run(self, tddft_dump: bool=True):
    """
        Args:
            TDDFT_dump: boolean

        Returns:
            set the do_tddft variable to one in cell.in
        """
    self.bse_tddft_options.update(do_bse='0', do_tddft='1' if tddft_dump else '0')