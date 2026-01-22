import re
from typing import List, Tuple, Union
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
def index_pattern(lines: List[str], pattern: str) -> int:
    repat = re.compile(pattern)
    for i, line in enumerate(lines):
        if repat.match(line):
            return i
    raise ValueError