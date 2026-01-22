import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def summary_writer_function(name, tensor, function, family=None):
    """Helper function to write summaries.

  Args:
    name: name of the summary
    tensor: main tensor to form the summary
    function: function taking a tag and a scope which writes the summary
    family: optional, the summary's family

  Returns:
    The result of writing the summary.
  """
    name_scope = ops.get_name_scope()
    if name_scope:
        name_scope += '/'

    def record():
        with ops.name_scope(name_scope), summary_op_util.summary_scope(name, family, values=[tensor]) as (tag, scope):
            with ops.control_dependencies([function(tag, scope)]):
                return constant_op.constant(True)
    if _summary_state.writer is None:
        return control_flow_ops.no_op()
    with ops.device('cpu:0'):
        op = smart_cond.smart_cond(_legacy_contrib_should_record_summaries(), record, _nothing, name='')
        if not context.executing_eagerly():
            ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)
    return op