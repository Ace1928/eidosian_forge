import collections as py_collections
import os
import pprint
import random
import sys
from absl import logging
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_logging_ops import *
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def get_summary_op():
    """Returns a single Summary op that would run all summaries.

  Either existing one from `SUMMARY_OP` collection or merges all existing
  summaries.

  Returns:
    If no summaries were collected, returns None. Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.
  """
    summary_op = ops.get_collection(ops.GraphKeys.SUMMARY_OP)
    if summary_op is not None:
        if summary_op:
            summary_op = summary_op[0]
        else:
            summary_op = None
    if summary_op is None:
        summary_op = merge_all_summaries()
        if summary_op is not None:
            ops.add_to_collection(ops.GraphKeys.SUMMARY_OP, summary_op)
    return summary_op