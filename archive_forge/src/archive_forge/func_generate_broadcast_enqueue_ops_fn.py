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
def generate_broadcast_enqueue_ops_fn(ctx, input_fn, inputs_structure_recorder, num_hosts):
    """Generates infeed enqueue ops for one input_fn on all the hosts."""
    captured_infeed_queue = _CapturedObject()
    dataset_initializer = None
    device_0 = ctx.tpu_host_placement_function(host_id=0)
    with tf.compat.v1.device(device_0):
        user_context = tpu_context.TPUContext(internal_ctx=ctx, input_device=device_0, invocation_index=0, host_id=0)
        inputs = _Inputs.from_input_fn(input_fn(user_context))
        is_dataset = inputs.is_dataset
        if ctx.mode == model_fn_lib.ModeKeys.PREDICT:
            if not is_dataset:
                raise TypeError('For mode PREDICT, `input_fn` must return `Dataset` instead of `features` and `labels`.')
            inputs = _InputsWithStoppingSignals(dataset=inputs.dataset, batch_size=ctx.batch_size_for_input_fn, add_padding=True)
        if is_dataset:
            dataset_initializer = inputs.dataset_initializer()
        num_replicas_per_host = ctx.num_of_replicas_per_host

    def tpu_ordinal_function_impl(shard_id):
        if ctx.device_assignment:
            return ctx.device_assignment.tpu_ordinal(replica=shard_id)
        else:
            return shard_id % num_replicas_per_host

    def device_function_impl(shard_id):
        return ctx.tpu_host_placement_function(replica_id=shard_id)

    def enqueue_ops_fn():
        """Generates enqueue ops for all the hosts."""
        broadcasted_inputs = []
        flattened_inputs = None
        signals = None
        num_replicas = ctx.num_replicas
        core_id = 0
        for host_id in xrange(num_hosts):
            with tf.compat.v1.device(ctx.tpu_host_placement_function(host_id=host_id)):
                for _ in xrange(ctx.num_of_replicas_per_host):
                    if flattened_inputs is None:
                        features, labels = inputs.features_and_labels()
                        signals = inputs.signals()
                        inputs_structure_recorder.validate_and_record_structure(features, labels)
                        flattened_inputs = inputs_structure_recorder.flatten_features_and_labels(features, labels, signals)
                        if ctx.config.tpu_config.eval_training_input_configuration is tpu_config.InputPipelineConfig.SLICED:
                            input_slices = [tf.split(x, num_replicas) for x in flattened_inputs]
                    if ctx.config.tpu_config.eval_training_input_configuration is tpu_config.InputPipelineConfig.SLICED:
                        broadcasted_inputs.append([x[core_id] for x in input_slices])
                        core_id += 1
                    else:
                        broadcasted_inputs.append(flattened_inputs)
        infeed_queue = tpu_feed.InfeedQueue(number_of_tuple_elements=len(broadcasted_inputs[0]))
        captured_infeed_queue.capture(infeed_queue)
        enqueue_ops = infeed_queue.generate_enqueue_ops(broadcasted_inputs, tpu_ordinal_function=tpu_ordinal_function_impl, placement_function=device_function_impl)
        if signals is None:
            return enqueue_ops
        else:
            return {'ops': enqueue_ops, 'signals': signals}
    return (enqueue_ops_fn, captured_infeed_queue, dataset_initializer)