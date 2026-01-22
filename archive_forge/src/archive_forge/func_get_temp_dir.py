from tensorflow.python.framework import test_util as _test_util
from tensorflow.python.platform import googletest as _googletest
from tensorflow.python.framework.test_util import assert_equal_graph_def
from tensorflow.python.framework.test_util import create_local_cluster
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import gpu_device_name
from tensorflow.python.framework.test_util import is_gpu_available
from tensorflow.python.ops.gradient_checker import compute_gradient_error
from tensorflow.python.ops.gradient_checker import compute_gradient
import functools
import sys
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['test.get_temp_dir'])
def get_temp_dir():
    """Returns a temporary directory for use during tests.

  There is no need to delete the directory after the test.

  @compatibility(TF2)
  This function is removed in TF2. Please use `TestCase.get_temp_dir` instead
  in a test case.
  Outside of a unit test, obtain a temporary directory through Python's
  `tempfile` module.
  @end_compatibility

  Returns:
    The temporary directory.
  """
    return _googletest.GetTempDir()