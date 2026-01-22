import collections
import copy
import enum
import re
import sys
import threading
import types
from typing import Any, AnyStr, Callable, List, NoReturn, Pattern, Tuple, Type, Union, Optional
from absl import app
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export
@classmethod
def from_node_def(cls, node_def, g, inputs=None, output_types=None, control_inputs=None, input_types=None, original_op=None, op_def=None):
    """Creates an `Operation`.

    NOTE: This constructor validates the name of the `Operation` (passed
    as `node_def.name`). Valid `Operation` names match the following
    regular expression:

        [A-Za-z0-9.][A-Za-z0-9_.\\\\-/]*

    Args:
      node_def: `node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`. Used for
        attributes of `node_def_pb2.NodeDef`, typically `name`, `op`, and
        `device`.  The `input` attribute is irrelevant here as it will be
        computed when generating the model.
      g: `Graph`. The parent graph.
      inputs: list of `Tensor` objects. The inputs to this `Operation`.
      output_types: list of `DType` objects.  List of the types of the `Tensors`
        computed by this operation.  The length of this list indicates the
        number of output endpoints of the `Operation`.
      control_inputs: list of operations or tensors from which to have a control
        dependency.
      input_types: List of `DType` objects representing the types of the tensors
        accepted by the `Operation`.  By default uses `[x.dtype.base_dtype for x
        in inputs]`.  Operations that expect reference-typed inputs must specify
        these explicitly.
      original_op: Optional. Used to associate the new `Operation` with an
        existing `Operation` (for example, a replica with the op that was
        replicated).
      op_def: Optional. The `op_def_pb2.OpDef` proto that describes the op type
        that this `Operation` represents.

    Raises:
      TypeError: if control inputs are not Operations or Tensors,
        or if `node_def` is not a `NodeDef`,
        or if `g` is not a `Graph`,
        or if `inputs` are not tensors,
        or if `inputs` and `input_types` are incompatible.
      ValueError: if the `node_def` name is not valid.

    Returns:
      Operation object.
    """
    if not isinstance(g, Graph):
        raise TypeError(f'Argument g must be a Graph. Received an instance of type {type(g)}')
    if not isinstance(node_def, node_def_pb2.NodeDef):
        raise TypeError(f'Argument node_def must be a NodeDef. Received an instance of type: {type(node_def)}.')
    if node_def.ByteSize() >= 1 << 31 or node_def.ByteSize() < 0:
        raise ValueError(f'Cannot create a tensor proto whose content is larger than 2GB. Size of tensor is {node_def.ByteSize()} bytes.')
    if not _VALID_OP_NAME_REGEX.match(node_def.name):
        raise ValueError(f'`{node_def.name}` is not a valid node name. Accepted names conform to Regex /{_VALID_OP_NAME_REGEX}/')
    del output_types
    if inputs is None:
        inputs = []
    elif not isinstance(inputs, list):
        raise TypeError(f'Argument inputs shall be a list of Tensors. Received an instance of type {type(inputs)}')
    for a in inputs:
        if not isinstance(a, tensor_lib.Tensor):
            raise TypeError(f'Items of argument inputs shall be Tensor. Received an instance of type {type(a)}.')
    if input_types is None:
        input_types = [i.dtype.base_dtype for i in inputs]
    elif not all((x.is_compatible_with(i.dtype) for i, x in zip(inputs, input_types))):
        raise TypeError("In op '%s', input types (%s) are not compatible with expected types (%s)" % (node_def.name, [i.dtype for i in inputs], input_types))
    control_input_ops = []
    if control_inputs:
        for c in control_inputs:
            control_op = None
            if isinstance(c, Operation):
                control_op = c
            elif isinstance(c, (tensor_lib.Tensor, internal.IndexedSlices)):
                control_op = c.op
            else:
                raise TypeError(f'Control input must be an Operation, a Tensor, or IndexedSlices. Received an instance of type {type(c)}.')
            control_input_ops.append(control_op)
    c_op = _create_c_op(g, node_def, inputs, control_input_ops, op_def=op_def)
    self = Operation(c_op, SymbolicTensor)
    self._init(g)
    self._original_op = original_op
    self._control_flow_post_processing(input_tensors=inputs)
    return self