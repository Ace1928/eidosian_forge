from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import compat
def get_resource_handle_data(graph_op):
    assert isinstance(graph_op, core.Symbol) and (not isinstance(graph_op, core.Value))
    with graph_op.graph._c_graph.get() as c_graph:
        handle_data = pywrap_tf_session.GetHandleShapeAndType(c_graph, graph_op._as_tf_output())
    return cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData.FromString(compat.as_bytes(handle_data))