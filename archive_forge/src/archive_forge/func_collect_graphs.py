import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
@contextlib.contextmanager
def collect_graphs(optimized=True):
    """Collects a flat list of pre- or post-optimization graphs.

  The collected graphs include device placements, which can be useful for
  testing.

  Usage:

  ```
  @def_function.function
  def f(x):
    return x + constant_op.constant(1.)

  with context.collect_graphs() as graphs:
    with ops.device("CPU:0"):
      f(constant_op.constant(1.))

  graph, = graphs  # `graph` contains a single GraphDef for inspection
  ```

  Args:
    optimized: whether to collect optimized graphs or non-optimized graphs

  Yields:
    A list of GraphDefs, populated when the context manager exits.
  """
    ctx = context()
    ctx.enable_graph_collection()
    try:
        graphs = []
        yield graphs
        metadata = ctx.export_run_metadata()
    finally:
        ctx.disable_graph_collection()
    for graph in metadata.function_graphs:
        if optimized:
            graphs.append(graph.post_optimization_graph)
        else:
            graphs.append(graph.pre_optimization_graph)