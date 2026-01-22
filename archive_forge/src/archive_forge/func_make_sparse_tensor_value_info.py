import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_sparse_tensor_value_info(name: str, elem_type: int, shape: Optional[Sequence[Union[str, int, None]]], doc_string: str='', shape_denotation: Optional[List[str]]=None) -> ValueInfoProto:
    """Makes a SparseTensor ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string
    sparse_tensor_type_proto = make_sparse_tensor_type_proto(elem_type, shape, shape_denotation)
    value_info_proto.type.sparse_tensor_type.CopyFrom(sparse_tensor_type_proto.sparse_tensor_type)
    return value_info_proto