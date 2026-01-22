import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, helper, subbyte
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
def unpack_int4(data: Union[np.int32, np.ndarray], dims: Union[int, Sequence[int]], signed: bool) -> np.ndarray:
    """Converts ndarray of int4 (as packed uint8) to f32
    See :ref:`onnx-detail-int4` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is
            None.
        dims: The dimensions are used to reshape the unpacked buffer
        signed: Whether the 4 bit integer is signed or unsigned

    Returns:
        A numpy array of float32 reshaped to dims.
    """
    single_func = lambda x: subbyte.unpack_single_4bitx2(x, signed)
    func = np.frompyfunc(single_func, 1, 2)
    res_high, res_low = func(data.ravel())
    res = np.empty((res_high.size + res_low.size,), dtype=np.float32)
    res[0::2] = res_high
    res[1::2] = res_low
    if res.size == np.prod(dims) + 1:
        res = res.ravel()[:-1]
    res = res.reshape(dims)
    return res