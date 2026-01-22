import numpy as np
from .... import symbol
from .... import ndarray as nd
from ....base import string_types
from ._import_helper import _convert_map as convert_map
def get_graph_metadata(self, graph):
    """
        Get the model metadata from a given onnx graph.
        """
    _params = set()
    for tensor_vals in graph.initializer:
        _params.add(tensor_vals.name)
    input_data = []
    for graph_input in graph.input:
        if graph_input.name not in _params:
            shape = [val.dim_value for val in graph_input.type.tensor_type.shape.dim]
            dtype = graph_input.type.tensor_type.elem_type
            input_data.append((graph_input.name, tuple(shape), dtype))
    output_data = []
    for graph_out in graph.output:
        shape = [val.dim_value for val in graph_out.type.tensor_type.shape.dim]
        output_data.append((graph_out.name, tuple(shape)))
    metadata = {'input_tensor_data': input_data, 'output_tensor_data': output_data}
    return metadata