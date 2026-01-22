import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import numpy as np
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def select_cross_device_ops(devices, session_config=None):
    """Find the best `CrossDeviceOps` locally given a `tf.compat.v1.ConfigProto`.

  Args:
    devices: a list of devices passed to `tf.distribute.Strategy`.
    session_config: a `tf.compat.v1.ConfigProto` or `None`. If `None`, it will
      make decision based on all logical devices.

  Returns:
    A subclass of `CrossDeviceOps`.
  """
    requested_devices = set((device_util.canonicalize(d) for d in devices))
    if ops.executing_eagerly_outside_functions():
        logical_gpus = context.context().list_logical_devices(device_type='GPU')
        physical_gpus = context.context().list_physical_devices(device_type='GPU')
        if len(logical_gpus) != len(physical_gpus):
            logging.warning('NCCL is not supported when using virtual GPUs, fallingback to reduction to one device')
            return ReductionToOneDevice()
        machine_devices = context.context().list_logical_devices()
    else:
        machine_devices = device_lib.list_local_devices(session_config=session_config)
    using_devices = set()
    for d in machine_devices:
        if device_util.canonicalize(d.name) in requested_devices:
            using_devices.add(d.name)
    if len(using_devices) != len(requested_devices):
        logging.warning('Some requested devices in `tf.distribute.Strategy` are not visible to TensorFlow: %s', ','.join(list(requested_devices - using_devices)))
    if any(('gpu' not in d.lower() for d in requested_devices)):
        logging.warning('There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.')
        return ReductionToOneDevice()
    if kernels.get_registered_kernels_for_op('NcclAllReduce'):
        return NcclAllReduce(num_packs=1)
    else:
        logging.warning('Nccl kernel is not found, not using nccl allreduce.')
        return ReductionToOneDevice()