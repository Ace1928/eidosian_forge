import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def verify_tf_loop_vars(init_vars, iter_entry_vars, iter_exit_vars, symbol_names, opts, check_shapes=True):
    """Verifies loop variables for consistency."""
    if check_shapes and 'shape_invariants' in opts:
        shape_invariants = opts['shape_invariants']
    else:
        shape_invariants = nest.map_structure(lambda _: None, iter_entry_vars)
    assert len(symbol_names) == len(shape_invariants)
    assert len(symbol_names) == len(init_vars)
    assert len(symbol_names) == len(iter_entry_vars)
    assert len(symbol_names) == len(iter_exit_vars)
    for i in range(len(symbol_names)):
        name = symbol_names[i]
        init = init_vars[i]
        entry = iter_entry_vars[i]
        exit_ = iter_exit_vars[i]
        invariant = shape_invariants[i]
        try:
            nest.assert_same_structure(init, entry, expand_composites=True)
        except (ValueError, TypeError):
            try:
                init_tensors = variable_utils.convert_variables_to_tensors(init)
                nest.assert_same_structure(init_tensors, entry, expand_composites=True)
            except (ValueError, TypeError) as e:
                raise TypeError("'{}' does not have the same nested structure after one iteration.\n\n{}".format(name, e)) from e
        try:
            nest.assert_same_structure(entry, exit_, expand_composites=True)
        except (ValueError, TypeError) as e:
            raise TypeError("'{}' does not have the same nested structure after one iteration.\n\n{}".format(name, e)) from e
        if invariant is not None:
            try:
                nest.assert_same_structure(init, invariant, expand_composites=False)
            except (ValueError, TypeError) as e:
                raise TypeError("'{}' does not have the same nested structure as its corresponding shape invariant.\n\n{}".format(name, e)) from e
        nest.map_structure(functools.partial(_verify_single_loop_var, name, check_shapes), init, entry, exit_, invariant)