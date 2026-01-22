import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_train() -> None:
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    training_mode = 1
    y, output_mean, output_var = _batchnorm_training_mode(x, s, bias, mean, var)
    node = onnx.helper.make_node('BatchNormalization', inputs=['x', 's', 'bias', 'mean', 'var'], outputs=['y', 'output_mean', 'output_var'], training_mode=training_mode)
    expect(node, inputs=[x, s, bias, mean, var], outputs=[y, output_mean, output_var], name='test_batchnorm_example_training_mode')
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    training_mode = 1
    momentum = 0.9
    epsilon = 0.01
    y, output_mean, output_var = _batchnorm_training_mode(x, s, bias, mean, var, momentum, epsilon)
    node = onnx.helper.make_node('BatchNormalization', inputs=['x', 's', 'bias', 'mean', 'var'], outputs=['y', 'output_mean', 'output_var'], epsilon=epsilon, training_mode=training_mode)
    expect(node, inputs=[x, s, bias, mean, var], outputs=[y, output_mean, output_var], name='test_batchnorm_epsilon_training_mode')