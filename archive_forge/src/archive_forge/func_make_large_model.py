from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def make_large_model(graph: onnx.GraphProto, large_initializers: dict[str, np.ndarray] | None=None, **kwargs: Any) -> ModelContainer:
    """Construct a ModelContainer

    C API and Python API of protobuf do not operate without serializing
    the protos. This function uses the Python API of ModelContainer.

    Arguments:
        graph: *make_graph* returns
        large_initializers: dictionary `name: large tensor`,
            large tensor is any python object supporting the DLPack protocol,
            the ownership the tensor is transferred to the ModelContainer,
            the tensor must define method `tobytes` like numpy tensors
        **kwargs: any attribute to add to the returned instance

    Returns:
        ModelContainer
    """
    model = onnx.helper.make_model(graph, **kwargs)
    large_model = ModelContainer()
    large_model.model_proto = model
    if large_initializers:
        large_model.set_large_initializers(large_initializers)
        large_model.check_large_initializers()
    return large_model