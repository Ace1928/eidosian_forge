import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
def map_vector_type_stubs_to_alias(vector_type_stubs):
    """For each of the stubs, create its aliases.

    For example: float64x3 -> double3
    """
    base_type_to_alias = {'char': f'int{np.dtype(np.byte).itemsize * 8}', 'short': f'int{np.dtype(np.short).itemsize * 8}', 'int': f'int{np.dtype(np.intc).itemsize * 8}', 'long': f'int{np.dtype(np.int_).itemsize * 8}', 'longlong': f'int{np.dtype(np.longlong).itemsize * 8}', 'uchar': f'uint{np.dtype(np.ubyte).itemsize * 8}', 'ushort': f'uint{np.dtype(np.ushort).itemsize * 8}', 'uint': f'uint{np.dtype(np.uintc).itemsize * 8}', 'ulong': f'uint{np.dtype(np.uint).itemsize * 8}', 'ulonglong': f'uint{np.dtype(np.ulonglong).itemsize * 8}', 'float': f'float{np.dtype(np.single).itemsize * 8}', 'double': f'float{np.dtype(np.double).itemsize * 8}'}
    base_type_to_vector_type = defaultdict(list)
    for stub in vector_type_stubs:
        base_type_to_vector_type[stub.__name__[:-2]].append(stub)
    for alias, base_type in base_type_to_alias.items():
        vector_type_stubs = base_type_to_vector_type[base_type]
        for stub in vector_type_stubs:
            nelem = stub.__name__[-1]
            stub.aliases.append(f'{alias}{nelem}')