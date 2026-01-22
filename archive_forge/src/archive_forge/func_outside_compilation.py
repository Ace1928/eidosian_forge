from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['tpu.outside_compilation'])
def outside_compilation(computation: Callable[..., Any], *args, **kwargs) -> Any:
    """Builds part of a computation outside any current TPU replicate scope.

  `tf.tpu.outside_compilation()` is used to run ops in `computation` on CPU
  instead of running on TPU. For example, users can run ops that are not
  supported on TPU's (e.g. tf.summary.write()) by explicitly placing those
  ops on CPU's. Below usage of outside compilation will place ops in
  `computation_with_string_ops` on CPU.

  Example usage:

  ```python
  def computation_with_string_ops(x):
    # strings types are not supported on TPU's and below ops must
    # run on CPU instead.
    output = tf.strings.format('1{}', x)
    return tf.strings.to_number(output)

  def tpu_computation():
    # Expected output is 11.
    output = tf.tpu.outside_compilation(computation_with_string_ops, 1)
  ```

  Outside compilation should be called inside TPUReplicateContext. That is,
  `tf.tpu.outside_compilation()` should be called inside a function that is
  passed to `tpu.split_compile_and_replicate()` -- this is implied when
  outside compilation is invoked inside a function passed to TPUStrategy
  `run()`. If invoked outside of TPUReplicateContext,
  then this simply returns the result of `computation`, and therefore,
  would be a no-op. Note that outside compilation is different from
  `tf.distribute.experimental.TPUStrategy.merge_call()` as logic in
  outside compilation is replicated and executed separately for each
  replica. On the other hand, `merge_call()` requires a `merge_fn`
  to aggregate the inputs from different replicas and is executed only
  once.

  For variables placed in TPU device, which includes variables created inside
  TPUStrategy scope, outside compilation logic must not include variable
  read/write. For variables placed on host, which is the case when variables
  created via TPUEstimator, variable read/write is only allowed if the variable
  is not accessed by any other ops in the TPU computation. Variable read/write
  from outside compilation cluster is not visible from TPU computation and
  vice versa. Therefore, if outside compilation logic contains such host
  variables read/write ops and if the variables are accessed by TPU
  computation as well, then this may lead to deadlock.

  Internally, `tf.tpu.outside_compilation()` adds outside compilation
  attributes to all ops in `computation`. During a later passes ops with outside
  compilation attributes are moved to a host-side graph. Inputs to this extract
  host-side graph are sent from TPU computation graph to host graph via a pair
  of XlaSendToHost and XlaRecvFromHost ops. Note that using
  `tf.tpu.outside_compilation()` may result in tensor transfer between TPU and
  CPU, leading to non-trivial performance impact.

  Args:
    computation: A Python function that builds the computation to place on the
      host.
    *args: the positional arguments for the computation.
    **kwargs: the keyword arguments for the computation.

  Returns:
    The Tensors returned by computation.
  """
    return outside_compilation_impl(False, computation, *args, **kwargs)