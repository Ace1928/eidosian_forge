import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
def make_vector_type_stubs():
    """Make user facing objects for vector types"""
    vector_type_stubs = []
    vector_type_prefix = ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64')
    vector_type_element_counts = (1, 2, 3, 4)
    vector_type_attribute_names = ('x', 'y', 'z', 'w')
    for prefix, nelem in itertools.product(vector_type_prefix, vector_type_element_counts):
        type_name = f'{prefix}x{nelem}'
        attr_names = vector_type_attribute_names[:nelem]
        vector_type_stub = type(type_name, (Stub,), {**{attr: lambda self: None for attr in attr_names}, **{'_description_': f'<{type_name}>', '__signature__': Signature(parameters=[Parameter(name=attr_name, kind=Parameter.POSITIONAL_ONLY) for attr_name in attr_names[:nelem]]), '__doc__': f'A stub for {type_name} to be used in CUDA kernels.'}, **{'aliases': []}})
        vector_type_stubs.append(vector_type_stub)
    return vector_type_stubs