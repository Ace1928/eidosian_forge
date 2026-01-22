from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def graph_inputs(op):
    return [x.op for x in op.inputs] + list(op.control_inputs)