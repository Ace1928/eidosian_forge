import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def remove_external_data_field(tensor: TensorProto, field_key: str) -> None:
    """Removes a field from a Tensor's external_data key-value store.

    Modifies tensor object in place.

    Arguments:
        tensor (TensorProto): Tensor object from which value will be removed
        field_key (string): The key of the field to be removed
    """
    for i, field in enumerate(tensor.external_data):
        if field.key == field_key:
            del tensor.external_data[i]