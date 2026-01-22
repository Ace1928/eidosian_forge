import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def graph_execution_traces(self, digest=False, begin=None, end=None):
    """Get all the intra-graph execution tensor traces read so far.

    Args:
      digest: Whether the results will be returned in the more light-weight
        digest form.
      begin: Optional beginning index for the requested traces or their digests.
        Python-style negative indices are supported.
      end: Optional ending index for the requested traces or their digests.
        Python-style negative indices are supported.

    Returns:
      If `digest`: a `list` of `GraphExecutionTraceDigest` objects.
      Else: a `list` of `GraphExecutionTrace` objects.
    """
    digests = self._graph_execution_trace_digests
    if begin is not None or end is not None:
        begin = begin or 0
        end = end or len(digests)
        digests = digests[begin:end]
    if digest:
        return digests
    else:
        return [self.read_graph_execution_trace(digest) for digest in digests]