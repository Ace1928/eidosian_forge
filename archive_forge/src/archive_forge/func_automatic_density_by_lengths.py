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
def automatic_density_by_lengths(cls, structure: Structure, length_densities: Sequence[float], force_gamma: bool=False) -> Self:
    """
        Returns an automatic Kpoint object based on a structure and a k-point
        density normalized by lattice constants.

        Algorithm:
            For a given dimension, the # of k-points is chosen as
            length_density = # of kpoints * lattice constant, e.g. [50.0, 50.0, 1.0] would
            have k-points of 50/a x 50/b x 1/c.

        Args:
            structure (Structure): Input structure
            length_densities (list[floats]): Defines the density of k-points in each
            dimension, e.g. [50.0, 50.0, 1.0].
            force_gamma (bool): Force a gamma centered mesh

        Returns:
            Kpoints
        """
    if len(length_densities) != 3:
        msg = f'The dimensions of length_densities must be 3, not {len(length_densities)}'
        raise ValueError(msg)
    comment = f'k-point density of {length_densities}/[a, b, c]'
    lattice = structure.lattice
    abc = lattice.abc
    num_div = [np.ceil(ld / abc[idx]) for idx, ld in enumerate(length_densities)]
    is_hexagonal = lattice.is_hexagonal()
    is_face_centered = structure.get_space_group_info()[0][0] == 'F'
    has_odd = any((idx % 2 == 1 for idx in num_div))
    if has_odd or is_hexagonal or is_face_centered or force_gamma:
        style = Kpoints.supported_modes.Gamma
    else:
        style = Kpoints.supported_modes.Monkhorst
    return cls(comment, 0, style, [num_div], (0, 0, 0))