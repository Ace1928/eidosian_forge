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
@property
def md5_header_hash(self) -> str:
    """Computes a md5 hash of the metadata defining the PotcarSingle."""
    hash_str = ''
    for k, v in self.keywords.items():
        if k in ('nentries', 'Orbitals', 'SHA256', 'COPYR'):
            continue
        hash_str += f'{k}'
        if isinstance(v, (bool, int)):
            hash_str += f'{v}'
        elif isinstance(v, float):
            hash_str += f'{v:.3f}'
        elif isinstance(v, (tuple, list)):
            for item in v:
                if isinstance(item, float):
                    hash_str += f'{item:.3f}'
                elif isinstance(item, (Orbital, OrbitalDescription)):
                    for item_v in item:
                        if isinstance(item_v, (int, str)):
                            hash_str += f'{item_v}'
                        elif isinstance(item_v, float):
                            hash_str += f'{item_v:.3f}'
                        else:
                            hash_str += f'{item_v}' if item_v else ''
        else:
            hash_str += v.replace(' ', '')
    self.hash_str = hash_str
    md5 = hashlib.new('md5', usedforsecurity=False)
    md5.update(hash_str.lower().encode('utf-8'))
    return md5.hexdigest()