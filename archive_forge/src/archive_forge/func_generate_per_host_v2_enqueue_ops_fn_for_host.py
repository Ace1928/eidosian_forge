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
def generate_per_host_v2_enqueue_ops_fn_for_host(ctx, input_fn, inputs_structure_recorder, device, host_id, invocation_index):
    """Generates infeed enqueue ops for per-host input_fn on a single host."""
    captured_infeed_queue = _CapturedObject()
    dataset_initializer = None
    with tf.compat.v1.device(device):
        user_context = tpu_context.TPUContext(internal_ctx=ctx, input_device=device, invocation_index=invocation_index, host_id=host_id)
        inputs = _Inputs.from_input_fn(input_fn(user_context))
        is_dataset = inputs.is_dataset
        if not is_dataset:
            raise TypeError('`input_fn` must return a `Dataset` for the PER_HOST_V2 input pipeline configuration.')
        if ctx.mode == model_fn_lib.ModeKeys.PREDICT:
            inputs = _InputsWithStoppingSignals(dataset=inputs.dataset, batch_size=ctx.batch_size_for_input_fn, add_padding=True, num_invocations_per_step=max(1, ctx.num_of_replicas_per_host))
        dataset_initializer = inputs.dataset_initializer()
        tpu_ordinal_function_impl = ctx.tpu_ordinal_function(host_id)

        def device_function_impl(shard_id):
            if ctx.device_assignment is not None:
                replica_id = ctx.device_assignment.lookup_replicas(task_id=host_id, logical_core=0)[shard_id]
                return ctx.tpu_host_placement_function(replica_id=replica_id)
            else:
                return None

    def enqueue_ops_fn():
        """Generates the per_host enqueue ops."""
        control_deps = []
        per_host_sharded_inputs = []
        enqueue_datas_list = []
        num_replicas_per_host = max(1, ctx.num_of_replicas_per_host)
        cached_signals = None
        with tf.compat.v1.device(device):
            if not inputs.is_dataset:
                raise TypeError('`input_fn` must return a `Dataset` for this mode.')
            for host in range(num_replicas_per_host):
                if ctx.allow_per_host_v2_parallel_get_next:
                    features, labels = inputs.features_and_labels()
                with tf.control_dependencies(control_deps):
                    if not ctx.allow_per_host_v2_parallel_get_next:
                        features, labels = inputs.features_and_labels()
                    signals = inputs.signals()
                    if cached_signals:
                        signals['stopping'] = cached_signals['stopping']
                    else:
                        cached_signals = signals
                features, labels, enqueue_data = _tpu_estimator_embedding.split_inputs(ctx, features, labels)
                if len(enqueue_data) != 1:
                    raise RuntimeError('Missing or extra enqueue_data for host {}. len(enqueue_data) = {}.'.format(host, len(enqueue_data)))
                enqueue_datas_list.append(enqueue_data[0])
                inputs_structure_recorder.validate_and_record_structure(features, labels)
                flattened_inputs = inputs_structure_recorder.flatten_features_and_labels(features, labels, signals)
                control_deps.extend(flattened_inputs)
                per_host_sharded_inputs.append(flattened_inputs)
            if inputs_structure_recorder.flattened_input_dims:
                input_partition_dims = inputs_structure_recorder.flattened_input_dims
                if signals:
                    input_partition_dims += [None] * len(signals)
                infeed_queue = tpu_feed._PartitionedInfeedQueue(number_of_tuple_elements=len(per_host_sharded_inputs[0]), host_id=host_id, input_partition_dims=input_partition_dims, device_assignment=ctx.device_assignment)
                per_host_enqueue_ops = infeed_queue.generate_enqueue_ops(per_host_sharded_inputs)
            else:
                infeed_queue = tpu_feed.InfeedQueue(number_of_tuple_elements=len(per_host_sharded_inputs[0]))
                per_host_enqueue_ops = infeed_queue.generate_enqueue_ops(per_host_sharded_inputs, tpu_ordinal_function=tpu_ordinal_function_impl, placement_function=device_function_impl)
            captured_infeed_queue.capture(infeed_queue)
        if ctx.embedding_config:
            per_host_enqueue_ops.extend(ctx.embedding_config.tpu_embedding.generate_enqueue_ops(enqueue_datas_list))
        if signals is None:
            return per_host_enqueue_ops
        else:
            return {'ops': per_host_enqueue_ops, 'signals': signals}
    return (enqueue_ops_fn, captured_infeed_queue, dataset_initializer)