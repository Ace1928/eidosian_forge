from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def scan_body(scan_state, scan_inputs):
    """Main body of the Dataset.scan."""
    loop_vars, iterate = (scan_state, scan_inputs)
    set_state(loop_vars)

    def main_path():
        body(iterate)
        new_loop_vars = get_state()
        control_flow.verify_tf_loop_vars(init_vars, loop_vars, new_loop_vars, symbol_names, opts, check_shapes=False)
        return new_loop_vars
    if extra_test is not None:
        extra_cond = extra_test()
        new_loop_vars = cond.cond(extra_cond, main_path, lambda: loop_vars)
    else:
        extra_cond = (constant_op.constant(True),)
        new_loop_vars = main_path()
    scan_outputs = (new_loop_vars, extra_cond)
    new_scan_state = new_loop_vars
    return (new_scan_state, scan_outputs)