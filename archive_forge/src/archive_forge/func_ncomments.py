from __future__ import annotations
import os
import re
import shutil
import warnings
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import __version__ as CURRENT_VER
from pymatgen.io.core import InputFile
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.template import TemplateInputGen
@property
def ncomments(self) -> int:
    """
        Returns the number of comments in the current LammpsInputFile. Includes the blocks of comments as well
        as inline comments (comment lines within blocks of LAMMPS commands).
        """
    n_comments = 0
    for stage in self.stages:
        if all((cmd.strip().startswith('#') for cmd, args in stage['commands'])):
            n_comments += 1
        else:
            n_comments += sum((1 for cmd, _args in stage['commands'] if cmd.strip().startswith('#')))
    return n_comments