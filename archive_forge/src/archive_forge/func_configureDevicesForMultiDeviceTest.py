import os
import random
import re
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import test_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
def configureDevicesForMultiDeviceTest(self, num_devices):
    """Configures number of logical devices for multi-device tests.

    It returns a list of device names. If invoked in GPU-enabled runtime, the
    last device name will be for a GPU device. Otherwise, all device names will
    be for a CPU device.

    Args:
      num_devices: The number of devices to configure.

    Returns:
      A list of device names to use for a multi-device test.
    """
    cpus = config.list_physical_devices('CPU')
    gpus = config.list_physical_devices('GPU')
    config.set_logical_device_configuration(cpus[0], [context.LogicalDeviceConfiguration() for _ in range(num_devices)])
    devices = ['/device:CPU:' + str(i) for i in range(num_devices - 1)]
    if gpus:
        devices.append('/device:GPU:0')
    else:
        devices.append('/device:CPU:' + str(num_devices - 1))
    return devices