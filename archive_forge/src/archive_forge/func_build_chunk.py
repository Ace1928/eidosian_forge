from abc import ABC, abstractmethod
from typing import (Dict, Any, Sequence, TextIO, Iterator, Optional, Union,
import re
from warnings import warn
from pathlib import Path, PurePath
import numpy as np
import ase
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import ParseError, read
from ase.io.utils import ImageChunk
from ase.calculators.singlepoint import SinglePointDFTCalculator, SinglePointKPoint
def build_chunk(fd: TextIO) -> _CHUNK:
    """Build chunk which contains 1 complete atoms object"""
    lines = []
    while True:
        line = next(fd)
        lines.append(line)
        if _OUTCAR_SCF_DELIM in line:
            for _ in range(4):
                lines.append(next(fd))
            break
    return lines