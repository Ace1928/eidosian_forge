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
def o2b(obj: Any, parts: List[bytes]):
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, dict):
        return {key: o2b(value, parts) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [o2b(value, parts) for value in obj]
    if isinstance(obj, np.ndarray):
        assert obj.dtype != object, 'Cannot convert ndarray of type "object" to bytes.'
        offset = sum((len(part) for part in parts))
        if not np.little_endian:
            obj = obj.byteswap()
        parts.append(obj.tobytes())
        return {'__ndarray__': [obj.shape, obj.dtype.name, offset]}
    if isinstance(obj, complex):
        return {'__complex__': [obj.real, obj.imag]}
    objtype = getattr(obj, 'ase_objtype')
    if objtype:
        dct = o2b(obj.todict(), parts)
        dct['__ase_objtype__'] = objtype
        return dct
    raise ValueError('Objects of type {type} not allowed'.format(type=type(obj)))