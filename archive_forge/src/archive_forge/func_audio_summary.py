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
@deprecated('2016-11-30', 'Please switch to tf.summary.audio. Note that tf.summary.audio uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in.')
def audio_summary(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None):
    """Outputs a `Summary` protocol buffer with audio.

  This op is deprecated. Please switch to tf.summary.audio.
  For an explanation of why this op was deprecated, and information on how to
  migrate, look
  ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
  `sample_rate`.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

  Args:
    tag: A scalar `Tensor` of type `string`. Used to build the `tag` of the
      summary values.
    tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
      or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
    sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the
      signal in hertz.
    max_outputs: Max number of batch elements to generate audio for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
    with ops.name_scope(name, 'AudioSummary', [tag, tensor]) as scope:
        sample_rate = ops.convert_to_tensor(sample_rate, dtype=dtypes.float32, name='sample_rate')
        val = gen_logging_ops.audio_summary_v2(tag=tag, tensor=tensor, max_outputs=max_outputs, sample_rate=sample_rate, name=scope)
        _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
    return val