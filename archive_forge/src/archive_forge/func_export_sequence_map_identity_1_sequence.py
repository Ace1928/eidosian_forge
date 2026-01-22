import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_sequence_map_identity_1_sequence():
    body = onnx.helper.make_graph([onnx.helper.make_node('Identity', ['in0'], ['out0'])], 'seq_map_body', [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N'])], [onnx.helper.make_tensor_value_info('out0', onnx.TensorProto.FLOAT, ['M'])])
    node = onnx.helper.make_node('SequenceMap', inputs=['x'], outputs=['y'], body=body)
    x = [np.random.uniform(0.0, 1.0, 10).astype(np.float32) for _ in range(3)]
    y = x
    input_type_protos = [onnx.helper.make_sequence_type_proto(onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N']))]
    output_type_protos = [onnx.helper.make_sequence_type_proto(onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N']))]
    expect(node, inputs=[x], outputs=[y], input_type_protos=input_type_protos, output_type_protos=output_type_protos, name='test_sequence_map_identity_1_sequence')