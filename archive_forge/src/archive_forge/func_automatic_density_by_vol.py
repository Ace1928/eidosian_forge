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
def automatic_density_by_vol(cls, structure: Structure, kppvol: int, force_gamma: bool=False) -> Self:
    """
        Returns an automatic Kpoint object based on a structure and a kpoint
        density per inverse Angstrom^3 of reciprocal cell.

        Algorithm:
            Same as automatic_density()

        Args:
            structure (Structure): Input structure
            kppvol (int): Grid density per Angstrom^(-3) of reciprocal cell
            force_gamma (bool): Force a gamma centered mesh

        Returns:
            Kpoints
        """
    vol = structure.lattice.reciprocal_lattice.volume
    kppa = kppvol * vol * len(structure)
    return cls.automatic_density(structure, kppa, force_gamma=force_gamma)