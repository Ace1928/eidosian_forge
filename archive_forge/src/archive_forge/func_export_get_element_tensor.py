from typing import Any, Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_get_element_tensor() -> None:
    optional = np.array([1, 2, 3, 4]).astype(np.float32)
    tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.FLOAT, shape=[4])
    optional_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)
    node = onnx.helper.make_node('OptionalGetElement', inputs=['optional_input'], outputs=['output'])
    output = optional_get_element_reference_implementation(optional)
    expect(node, inputs=[optional], outputs=[output], input_type_protos=[optional_type_proto], name='test_optional_get_element_optional_tensor')
    expect(node, inputs=[optional], outputs=[output], input_type_protos=[tensor_type_proto], name='test_optional_get_element_tensor')