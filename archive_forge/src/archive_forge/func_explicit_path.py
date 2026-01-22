from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
@classmethod
def explicit_path(cls, ndivsm, kpath_bounds):
    """See _path for the meaning of the variables."""
    return cls._path(ndivsm, kpath_bounds=kpath_bounds, comment='Explicit K-path')