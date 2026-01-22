import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
def make_parent_line(p: Sequence[str]) -> str:
    return f'PARENT {' '.join(p)}'