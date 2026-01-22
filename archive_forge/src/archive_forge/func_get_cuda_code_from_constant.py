from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
def get_cuda_code_from_constant(x: Union[bool, int, float, complex], ctype: Scalar) -> str:
    dtype = ctype.dtype
    suffix_literal = _suffix_literals_dict.get(dtype.name)
    if suffix_literal is not None:
        s = str(x).lower()
        return f'{s}{suffix_literal}'
    ctype_str = str(ctype)
    if dtype.kind == 'c':
        return f'{ctype_str}({x.real}, {x.imag})'
    if ' ' in ctype_str:
        return f'({ctype_str}){x}'
    return f'{ctype_str}({x})'