import collections
import copy
import itertools
import json
import os
import typing
from absl import flags
from absl.testing import parameterized
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.config import is_gpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import is_tpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import preferred_device_type  # pylint: disable=unused-import
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.dtensor.python.tests.test_backend_name import DTensorTestUtilBackend
from tensorflow.dtensor.python.tests.test_backend_util import DTensorTestBackendConfigurator
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test as tf_test
@staticmethod
def configTestMesh(device_type_mesh_map: typing.Dict[typing.Text, layout_lib.Mesh]) -> layout_lib.Mesh:
    """Configs corresponding mesh given test context.

    If runs on a CPU mesh, set virtual device on CPU.
    If runs on a GPU mesh, sets virtual device on GPU with proper memory limits.
    if runs on a TPU mesh, initializes TPU system.

    Args:
      device_type_mesh_map: A dictionary containing device_type -> mesh mapping.

    Returns:
      A properly configured mesh for use in test.
    """
    reset_context()

    def get_mesh(device_type):
        mesh = device_type_mesh_map.get(device_type, None)
        if mesh is None:
            raise ValueError('Requires a %s mesh to run test on %s.' % (device_type, device_type))
        return mesh
    mesh = None
    if is_tpu_present():
        mesh = get_mesh('TPU')
        reset_context()
        accelerator_util.initialize_accelerator_system('TPU')
    elif tf_config.list_physical_devices('GPU'):
        mesh = get_mesh('GPU')
        reset_logical_devices('GPU', np.prod(mesh.shape()))
        accelerator_util.initialize_accelerator_system('GPU')
    else:
        mesh = get_mesh('CPU')
        reset_logical_devices('CPU', np.prod(mesh.shape()))
        accelerator_util.initialize_accelerator_system('CPU')
    test_backend_util.config_test_mesh(mesh)
    return mesh