from typing import Any, Tuple
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_seq_length() -> None:
    input = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]).astype(np.float32)
    input_size = 3
    hidden_size = 5
    node = onnx.helper.make_node('RNN', inputs=['X', 'W', 'R', 'B'], outputs=['', 'Y_h'], hidden_size=hidden_size)
    W = np.random.randn(1, hidden_size, input_size).astype(np.float32)
    R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32)
    W_B = np.random.randn(1, hidden_size).astype(np.float32)
    R_B = np.random.randn(1, hidden_size).astype(np.float32)
    B = np.concatenate((W_B, R_B), axis=1)
    rnn = RNNHelper(X=input, W=W, R=R, B=B)
    _, Y_h = rnn.step()
    expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_rnn_seq_length')