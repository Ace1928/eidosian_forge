from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def pcm_template(pcm: dict) -> str:
    """
        PCM run template.

        Args:
            pcm ():

        Returns:
            str: PCM template.
        """
    pcm_list = []
    pcm_list.append('$pcm')
    for key, value in pcm.items():
        pcm_list.append(f'   {key} {value}')
    pcm_list.append('$end')
    return '\n'.join(pcm_list)