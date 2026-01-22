import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def parse_path_string(s):
    """Parse compact string representation of BZ path.

    A path string can have several non-connected sections separated by
    commas. The return value is a list of sections where each section is a
    list of labels.

    Examples:

    >>> parse_path_string('GX')
    [['G', 'X']]
    >>> parse_path_string('GX,M1A')
    [['G', 'X'], ['M1', 'A']]
    """
    paths = []
    for path in s.split(','):
        names = [name for name in re.split('([A-Z][a-z0-9]*)', path) if name]
        paths.append(names)
    return paths