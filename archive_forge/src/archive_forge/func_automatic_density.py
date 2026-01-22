from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
@classmethod
def automatic_density(cls, structure: Structure, kppa: float, force_gamma: bool=False) -> Self:
    """
        Returns an automatic Kpoint object based on a structure and a kpoint
        density. Uses Gamma centered meshes for hexagonal cells and face-centered cells,
        Monkhorst-Pack grids otherwise.

        Algorithm:
            Uses a simple approach scaling the number of divisions along each
            reciprocal lattice vector proportional to its length.

        Args:
            structure (Structure): Input structure
            kppa (float): Grid density
            force_gamma (bool): Force a gamma centered mesh (default is to
                use gamma only for hexagonal cells or odd meshes)

        Returns:
            Kpoints
        """
    comment = f'pymatgen with grid density = {kppa:.0f} / number of atoms'
    if math.fabs(math.floor(kppa ** (1 / 3) + 0.5) ** 3 - kppa) < 1:
        kppa += kppa * 0.01
    lattice = structure.lattice
    lengths = lattice.abc
    ngrid = kppa / len(structure)
    mult = (ngrid * lengths[0] * lengths[1] * lengths[2]) ** (1 / 3)
    num_div = [int(math.floor(max(mult / length, 1))) for length in lengths]
    is_hexagonal = lattice.is_hexagonal()
    is_face_centered = structure.get_space_group_info()[0][0] == 'F'
    has_odd = any((idx % 2 == 1 for idx in num_div))
    if has_odd or is_hexagonal or is_face_centered or force_gamma:
        style = Kpoints.supported_modes.Gamma
    else:
        style = Kpoints.supported_modes.Monkhorst
    return cls(comment, 0, style, [num_div], (0, 0, 0))