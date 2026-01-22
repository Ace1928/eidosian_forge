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
def b2o(obj: Any, b: bytes) -> Any:
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, list):
        return [b2o(value, b) for value in obj]
    assert isinstance(obj, dict)
    x = obj.get('__complex__')
    if x is not None:
        return complex(*x)
    x = obj.get('__ndarray__')
    if x is not None:
        shape, name, offset = x
        dtype = np.dtype(name)
        size = dtype.itemsize * np.prod(shape).astype(int)
        a = np.frombuffer(b[offset:offset + size], dtype)
        a.shape = shape
        if not np.little_endian:
            a = a.byteswap()
        return a
    dct = {key: b2o(value, b) for key, value in obj.items()}
    objtype = dct.pop('__ase_objtype__', None)
    if objtype is None:
        return dct
    return create_ase_object(objtype, dct)