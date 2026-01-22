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
def gamma_automatic(cls, kpts: tuple[int, int, int]=(1, 1, 1), shift: Vector3D=(0, 0, 0)) -> Self:
    """
        Constructor for an automatic Gamma centered Kpoint grid.

        Args:
            kpts: Subdivisions N_1, N_2 and N_3 along reciprocal lattice
                vectors. Defaults to (1,1,1)
            shift: Shift to be applied to the kpoints. Defaults to (0,0,0).

        Returns:
            Kpoints object
        """
    return cls('Automatic kpoint scheme', 0, Kpoints.supported_modes.Gamma, kpts=[kpts], kpts_shift=shift)