import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def convert_model_from_external_data(model: ModelProto) -> None:
    """Call to set all tensors which use external data as embedded data.
    save_model saves all the tensors data as embedded data after
    calling this function.

    Arguments:
        model (ModelProto): Model to be converted.
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            if not tensor.HasField('raw_data'):
                raise ValueError("raw_data field doesn't exist.")
            del tensor.external_data[:]
            tensor.data_location = TensorProto.DEFAULT