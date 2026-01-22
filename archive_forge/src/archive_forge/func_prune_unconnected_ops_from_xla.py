import collections
import enum
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
import numpy as np
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def prune_unconnected_ops_from_xla(prune_graph: ops.Graph):
    """Prunes unconnected ops as listed in _UNCONNECTED_OPS_TO_PRUNE.

  Args:
    prune_graph: A tensorflow graph from which we wish to prune unconnected ops
      as listed in _UNCONNECTED_OPS_TO_PRUNE.  In general, these ops should have
      no inputs and no consumers. These can often be left behind due to graph
      construction rewiring (for instance TF-Hub). While they never execute,
      they will cause XLA compile to fail so we strip them from XLA compile by
      removing the tpu_replicate attribute.
  """
    for graph in [prune_graph] + [f for f in prune_graph._functions.values()]:
        if not isinstance(graph, ops.Graph):
            continue
        for op in graph.get_operations():
            if op.type not in _UNCONNECTED_OPS_TO_PRUNE:
                continue
            outputs_consumed = False
            for output in op.outputs:
                if output.consumers():
                    outputs_consumed = True
                    break
            if not outputs_consumed:
                logging.info('Pruning OP %s of type %s from XLA Compile due to it being disconnected.', op.name, op.type)
                op._clear_attr(tpu_replication._TPU_REPLICATE_ATTR)