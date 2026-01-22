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
def from_symbol_and_functional(cls, symbol: str, functional: str | None=None) -> Self:
    """Makes a PotcarSingle from a symbol and functional.

        Args:
            symbol (str): Symbol, e.g., Li_sv
            functional (str): Functional, e.g., PBE

        Returns:
            PotcarSingle
        """
    functional = functional or SETTINGS.get('PMG_DEFAULT_FUNCTIONAL', 'PBE')
    assert isinstance(functional, str)
    funcdir = cls.functional_dir[functional]
    PMG_VASP_PSP_DIR = SETTINGS.get('PMG_VASP_PSP_DIR')
    if PMG_VASP_PSP_DIR is None:
        raise ValueError(f'No POTCAR for {symbol} with functional={functional!r} found. Please set the PMG_VASP_PSP_DIR in .pmgrc.yaml.')
    paths_to_try = [os.path.join(PMG_VASP_PSP_DIR, funcdir, f'POTCAR.{symbol}'), os.path.join(PMG_VASP_PSP_DIR, funcdir, symbol, 'POTCAR')]
    for path in paths_to_try:
        path = os.path.expanduser(path)
        path = zpath(path)
        if os.path.isfile(path):
            return cls.from_file(path)
    raise OSError(f'You do not have the right POTCAR with functional={functional!r} and symbol={symbol!r} in your PMG_VASP_PSP_DIR={PMG_VASP_PSP_DIR!r}. Paths tried: {paths_to_try}')