from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_pcm_nonels(string: str) -> dict:
    """
        Read pcm_nonels parameters from string.

        Args:
            string (str): String

        Returns:
            dict[str, str]: PCM parameters
        """
    header = '^\\s*\\$pcm_nonels'
    row = '\\s*([a-zA-Z\\_]+)\\s+(.+)'
    footer = '^\\s*\\$end'
    pcm_nonels_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    if not pcm_nonels_table:
        print("No valid $pcm_nonels inputs found. Note that there should be no '=' characters in $pcm_nonels input lines.")
        return {}
    return dict(pcm_nonels_table[0])