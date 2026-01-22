import functools
import json
import numbers
import operator
import os
import re
import warnings
from time import time
from typing import List, Dict, Any
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, all_changes
from ase.data import atomic_numbers
from ase.db.row import AtomsRow
from ase.formula import Formula
from ase.io.jsonio import create_ase_object
from ase.parallel import world, DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock, PurePath
def object_to_bytes(obj: Any) -> bytes:
    """Serialize Python object to bytes."""
    parts = [b'12345678']
    obj = o2b(obj, parts)
    offset = sum((len(part) for part in parts))
    x = np.array(offset, np.int64)
    if not np.little_endian:
        x.byteswap(True)
    parts[0] = x.tobytes()
    parts.append(json.dumps(obj, separators=(',', ':')).encode())
    return b''.join(parts)