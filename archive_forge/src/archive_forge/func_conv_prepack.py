import importlib
import inspect
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration
def conv_prepack(g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups):
    output = g.op('_caffe2::WeightPrepack', input, weight, bias)
    symbolic_helper._quantized_ops.add(output)
    return output