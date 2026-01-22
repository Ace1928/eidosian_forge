import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def tf_output(c_op, index):
    """Returns a wrapped TF_Output with specified operation and index.

  Args:
    c_op: wrapped TF_Operation
    index: integer

  Returns:
    Wrapped TF_Output
  """
    ret = c_api.TF_Output()
    ret.oper = c_op
    ret.index = index
    return ret