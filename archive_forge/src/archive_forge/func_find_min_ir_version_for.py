import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def find_min_ir_version_for(opsetidlist: Sequence[OperatorSetIdProto], ignore_unknown: bool=False) -> int:
    """Given list of opset ids, determine minimum IR version required.

    Args:
        opsetidlist: A sequence of OperatorSetIdProto.
        ignore_unknown: If True, ignore unknown domain and return default minimum
            version for that domain.

    Returns:
        The minimum IR version required (integer)
    """
    default_min_version = 3

    def find_min(domain: Union[str, None], version: int) -> int:
        key = (domain or 'ai.onnx', version)
        if key in OP_SET_ID_VERSION_MAP:
            return OP_SET_ID_VERSION_MAP[key]
        if ignore_unknown:
            return default_min_version
        raise ValueError('Unsupported opset-version.')
    if opsetidlist:
        return max((find_min(x.domain, x.version) for x in opsetidlist))
    return default_min_version