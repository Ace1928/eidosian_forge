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
def get_coords(self) -> np.ndarray:
    """Return the fractional coordinate of atoms in structure.

        Returns:
            np.ndarray: Fractional coordinates of atoms of shape=(num_atoms*3,)
        """
    coords_lst = []
    aim_content = 'POSITION'
    aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
    for tmp_str in self.strs_lst[aim_idx + 1:aim_idx + self.num_atoms + 1]:
        tmp_strs_lst = tmp_str.split()
        tmp_coord = [float(value) for value in tmp_strs_lst[1:4]]
        coords_lst.append(tmp_coord)
    return np.array(coords_lst).reshape(-1)