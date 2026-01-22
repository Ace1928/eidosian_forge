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
@classmethod
def from_section(cls, section: Section) -> Self:
    """Extract GTH-formatted string from a section and convert it to model."""
    sec = copy.deepcopy(section)
    sec.verbosity(verbosity=False)
    lst = sec.get_str().split('\n')
    string = '\n'.join((line for line in lst if not line.startswith('&')))
    return cls.from_str(string)