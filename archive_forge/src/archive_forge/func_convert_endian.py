import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, helper, subbyte
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
def convert_endian(tensor: TensorProto) -> None:
    """Call to convert endianess of raw data in tensor.

    Args:
        tensor: TensorProto to be converted.
    """
    tensor_dtype = tensor.data_type
    np_dtype = helper.tensor_dtype_to_np_dtype(tensor_dtype)
    tensor.raw_data = np.frombuffer(tensor.raw_data, dtype=np_dtype).byteswap().tobytes()