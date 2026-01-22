import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN, ONNX_DOMAIN
@staticmethod
def export_gradient_scalar_add() -> None:
    add_node = onnx.helper.make_node('Add', ['a', 'b'], ['c'], name='my_add')
    gradient_node = onnx.helper.make_node('Gradient', ['a', 'b'], ['dc_da', 'dc_db'], name='my_gradient', domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN, xs=['a', 'b'], y='c')
    a = np.array(1.0).astype(np.float32)
    b = np.array(2.0).astype(np.float32)
    c = a + b
    dc_da = np.array(1).astype(np.float32)
    dc_db = np.array(1).astype(np.float32)
    graph = onnx.helper.make_graph(nodes=[add_node, gradient_node], name='GradientOfAdd', inputs=[onnx.helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT, []), onnx.helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT, [])], outputs=[onnx.helper.make_tensor_value_info('c', onnx.TensorProto.FLOAT, []), onnx.helper.make_tensor_value_info('dc_da', onnx.TensorProto.FLOAT, []), onnx.helper.make_tensor_value_info('dc_db', onnx.TensorProto.FLOAT, [])])
    opsets = [onnx.helper.make_operatorsetid(ONNX_DOMAIN, 12), onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)]
    model = onnx.helper.make_model_gen_version(graph, producer_name='backend-test', opset_imports=opsets)
    expect(model, inputs=[a, b], outputs=[c, dc_da, dc_db], name='test_gradient_of_add')