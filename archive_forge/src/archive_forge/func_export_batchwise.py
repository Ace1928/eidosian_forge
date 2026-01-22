from typing import Any, Tuple
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_batchwise() -> None:
    input = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)
    input_size = 2
    hidden_size = 4
    weight_scale = 0.5
    layout = 1
    node = onnx.helper.make_node('RNN', inputs=['X', 'W', 'R'], outputs=['Y', 'Y_h'], hidden_size=hidden_size, layout=layout)
    W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)
    rnn = RNNHelper(X=input, W=W, R=R, layout=layout)
    Y, Y_h = rnn.step()
    expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32)], name='test_simple_rnn_batchwise')