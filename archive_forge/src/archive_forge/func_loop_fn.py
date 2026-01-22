from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients_impl as gradient_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.util import nest
def loop_fn(i):
    y = array_ops.gather(output, i, axis=1)
    return gradient_ops.gradients(y, inp)[0]