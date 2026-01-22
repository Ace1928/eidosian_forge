from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def geom_opt_template(geom_opt: dict) -> str:
    """
        Args:
            geom_opt ():

        Returns:
            str: Geometry optimization section.
        """
    geom_opt_list = []
    geom_opt_list.append('$geom_opt')
    for key, value in geom_opt.items():
        geom_opt_list.append(f'   {key} = {value}')
    geom_opt_list.append('$end')
    return '\n'.join(geom_opt_list)