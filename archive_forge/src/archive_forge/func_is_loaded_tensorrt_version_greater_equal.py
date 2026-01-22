import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def is_loaded_tensorrt_version_greater_equal(major, minor=0, patch=0):
    ver = _pywrap_py_utils.get_loaded_tensorrt_version()
    return _is_tensorrt_version_greater_equal(ver, (major, minor, patch))