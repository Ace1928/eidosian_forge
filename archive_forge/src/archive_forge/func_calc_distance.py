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
def calc_distance(hsp1: str, hsp2: str) -> float:
    """Calculate the distance of two high symmetry points.

            Returns:
                distance (float): Calculate the distance of two high symmetry points. With factor of 2*pi.
            """
    hsp1_coord: np.ndarray = np.dot(np.array(self.kpath['kpoints'][hsp1]).reshape(1, 3), self.reciprocal_lattice)
    hsp2_coord: np.ndarray = np.dot(np.array(self.kpath['kpoints'][hsp2]).reshape(1, 3), self.reciprocal_lattice)
    return float(np.linalg.norm(hsp2_coord - hsp1_coord))