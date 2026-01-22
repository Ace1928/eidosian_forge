import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_clip_default_int8() -> None:
    node = onnx.helper.make_node('Clip', inputs=['x', 'min'], outputs=['y'])
    min_val = np.int8(0)
    x = np.random.randn(3, 4, 5).astype(np.int8)
    y = np.clip(x, min_val, np.iinfo(np.int8).max)
    expect(node, inputs=[x, min_val], outputs=[y], name='test_clip_default_int8_min')
    no_min = ''
    node = onnx.helper.make_node('Clip', inputs=['x', no_min, 'max'], outputs=['y'])
    max_val = np.int8(0)
    x = np.random.randn(3, 4, 5).astype(np.int8)
    y = np.clip(x, np.iinfo(np.int8).min, max_val)
    expect(node, inputs=[x, max_val], outputs=[y], name='test_clip_default_int8_max')
    no_max = ''
    node = onnx.helper.make_node('Clip', inputs=['x', no_min, no_max], outputs=['y'])
    x = np.array([-1, 0, 1]).astype(np.int8)
    y = np.array([-1, 0, 1]).astype(np.int8)
    expect(node, inputs=[x], outputs=[y], name='test_clip_default_int8_inbounds')