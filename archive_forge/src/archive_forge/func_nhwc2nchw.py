import importlib
import inspect
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration
def nhwc2nchw(g: jit_utils.GraphContext, input):
    axes = [0, 3, 1, 2]
    return _permute_helper(g, input, axes)