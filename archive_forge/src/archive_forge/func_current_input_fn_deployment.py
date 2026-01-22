from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
def current_input_fn_deployment(self):
    """The configuration of the current input_fn invocation.

    The configuration depends on `TPUConfig.per_host_input_for_training`. See
    `TPUConfig` for details.

    Only set in params dict of input_fn

    Returns:
      A tuple of
        1. Device spec string: String, is the current CPU host where the
           input_fn is invoked.
        2. Current invocation index: Int, 0-based index of the input_fn
           invocation. See next item for details.
        3. Total invocation count: Int, the total number of times to invoke the
           input_fn on all CPU hosts. Each invocation will be passed with a new
           `TPUContext` instance with current invocation index set properly.
        4. Total number of replicas consumed by current_invocation: Int, the
           number of replicas fed by the data returned by current input_fn. For
           example, for per_core input pipeline deployment
           and non-model-parallelism, total invocation count is equal to
           the number of cores in the system and num replicas consumed by
           current invocation is 1. For per-host v2 input pipeline deployment,
           total invocation count is equal to the number of hosts in the system
           and num replicas consumed by current invocation is equal to number of
           replicas per host.

    Raises:
      RuntimeError: If this method is not be called from input_fn.
    """
    if not self._call_from_input_fn:
        raise RuntimeError('This TPUContext instance must not be called from model_fn.')
    if self._internal_ctx.is_input_sharded_per_core():
        total_invocation_count = self._internal_ctx.num_hosts * self._internal_ctx.num_of_replicas_per_host
        replicas_consumed = 1
    elif self._internal_ctx.is_input_broadcast_with_iterators():
        total_invocation_count = 1
        replicas_consumed = self._internal_ctx.num_replicas
    elif self._internal_ctx.is_replica_across_hosts():
        total_invocation_count = self._internal_ctx.num_replicas
        replicas_consumed = 1
    else:
        total_invocation_count = self._internal_ctx.num_hosts
        replicas_consumed = self._internal_ctx.num_of_replicas_per_host
    return (self._input_device, self._invocation_index, total_invocation_count, replicas_consumed)