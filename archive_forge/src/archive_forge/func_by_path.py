from __future__ import annotations
import abc
import copy
import hashlib
import itertools
import os
import re
import textwrap
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.io.cp2k.utils import chunk, postprocessor, preprocessor
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def by_path(self, path: str):
    """
        Access a sub-section using a path. Used by the file parser.

        Args:
            path (str): Path to section of form 'SUBSECTION1/SUBSECTION2/SUBSECTION_OF_INTEREST'
        """
    _path = path.split('/')
    if _path[0].upper() == self.name.upper():
        _path = _path[1:]
    sec_str = self
    for p in _path:
        sec_str = sec_str.get_section(p)
    return sec_str