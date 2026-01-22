from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import enum
import math
import os
import signal
import sys
import threading
import time
import tensorflow as tf
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import preempted_hook
from tensorflow.python.tpu import session_support
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import evaluation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import error_handling
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_context
from tensorflow_estimator.python.estimator.tpu import util as util_lib
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdagradParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdamParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import EmbeddingConfigSpec  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import StochasticGradientDescentParameters  # pylint: disable=unused-import
def call_computation(computation_inputs, computation, batch_config=None):
    """Call computation.

  Args:
    computation_inputs: A tensor or dict of tensors, the inputs to the
      computation.
    computation: A Python function that takes no inputs and builds computation
      graph. If `computation` returns m outputs, this function will return a
      list of m Tensors.
    batch_config: A BatchConfig named tuple specifying the batching
      configuration to use for inference batching.

  Returns:
    A list of output tensors.
  """

    def tpu_partitioned_call(partition_inputs):

        @function.Defun(capture_resource_var_by_value=False)
        def tpu_subgraph():
            return computation(partition_inputs)
        return tpu_functional.TPUPartitionedCall(args=tpu_subgraph.captured_inputs, device_ordinal=tpu_ops.tpu_ordinal_selector(), Tout=[o.type for o in tpu_subgraph.definition.signature.output_arg], f=tpu_subgraph)
    if not batch_config:
        return tpu_partitioned_call(computation_inputs)
    ordered_inputs_list = tf.nest.flatten(computation_inputs)

    @tf.nondifferentiable_batch_function(num_batch_threads=batch_config.num_batch_threads, max_batch_size=batch_config.max_batch_size, batch_timeout_micros=batch_config.batch_timeout_micros, allowed_batch_sizes=batch_config.allowed_batch_sizes, max_enqueued_batches=batch_config.max_enqueued_batches, autograph=False)
    def batched_tpu_computation(*tensor_args):
        """Recompose the input feature dict and calls the TPU computation."""
        computation_feature_input = tf.nest.pack_sequence_as(computation_inputs, tensor_args)
        return tpu_partitioned_call(computation_feature_input)
    return batched_tpu_computation(*ordered_inputs_list)