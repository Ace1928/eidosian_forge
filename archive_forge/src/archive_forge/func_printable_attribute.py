import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def printable_attribute(attr: AttributeProto, subgraphs: bool=False) -> Union[str, Tuple[str, List[GraphProto]]]:
    content = []
    content.append(attr.name)
    content.append('=')

    def str_float(f: float) -> str:
        return f'{f:.15g}'

    def str_int(i: int) -> str:
        return str(i)
    _T = TypeVar('_T')

    def str_list(str_elem: Callable[[_T], str], xs: Sequence[_T]) -> str:
        return '[' + ', '.join(map(str_elem, xs)) + ']'
    graphs = []
    if attr.HasField('f'):
        content.append(str_float(attr.f))
    elif attr.HasField('i'):
        content.append(str_int(attr.i))
    elif attr.HasField('s'):
        content.append(repr(_sanitize_str(attr.s)))
    elif attr.HasField('t'):
        if len(attr.t.dims) > 0:
            content.append('<Tensor>')
        else:
            field = tensor_dtype_to_field(attr.t.data_type)
            content.append(f'<Scalar Tensor {getattr(attr.t, field)}>')
    elif attr.HasField('g'):
        content.append(f'<graph {attr.g.name}>')
        graphs.append(attr.g)
    elif attr.HasField('tp'):
        content.append(f'<Type Proto {attr.tp}>')
    elif attr.floats:
        content.append(str_list(str_float, attr.floats))
    elif attr.ints:
        content.append(str_list(str_int, attr.ints))
    elif attr.strings:
        content.append(str(list(map(_sanitize_str, attr.strings))))
    elif attr.tensors:
        content.append('[<Tensor>, ...]')
    elif attr.type_protos:
        content.append('[')
        for i, tp in enumerate(attr.type_protos):
            comma = ',' if i != len(attr.type_protos) - 1 else ''
            content.append(f'<Type Proto {tp}>{comma}')
        content.append(']')
    elif attr.graphs:
        content.append('[')
        for i, g in enumerate(attr.graphs):
            comma = ',' if i != len(attr.graphs) - 1 else ''
            content.append(f'<graph {g.name}>{comma}')
        content.append(']')
        graphs.extend(attr.graphs)
    else:
        content.append('<Unknown>')
    if subgraphs:
        return (' '.join(content), graphs)
    return ' '.join(content)