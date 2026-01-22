from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def plots_template(plots: dict) -> str:
    """
        Args:
            plots ():

        Returns:
            str: Plots section.
        """
    plots_list = []
    plots_list.append('$plots')
    for key, value in plots.items():
        plots_list.append(f'   {key} {value}')
    plots_list.append('$end')
    return '\n'.join(plots_list)