from __future__ import annotations
import linecache
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.kpath import KPathSeek
@staticmethod
def locate_all_lines(strs_lst: list[str], content: str) -> list[int]:
    """Locate the elements in list where a certain paragraph of text is located (return all indices)

        Args:
            strs_lst (list[str]): List of strings.
            content (str): Certain paragraph of text that needs to be located.
        """
    str_idxs: list[int] = []
    str_no: int = -1
    for tmp_str in strs_lst:
        str_no += 1
        if content.upper() in tmp_str.upper():
            str_idxs.append(str_no)
    return str_idxs