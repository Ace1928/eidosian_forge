from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import _op_def_library_pybind
from tensorflow.core.framework import attr_value_pb2
Helper method to speed up `_apply_op_helper` in op_def_library.