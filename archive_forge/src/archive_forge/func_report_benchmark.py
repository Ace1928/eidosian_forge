import math
import numbers
import os
import re
import sys
import time
import types
from absl import app
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def report_benchmark(self, iters=None, cpu_time=None, wall_time=None, throughput=None, extras=None, name=None, metrics=None):
    """Report a benchmark.

    Args:
      iters: (optional) How many iterations were run
      cpu_time: (optional) Median or mean cpu time in seconds.
      wall_time: (optional) Median or mean wall time in seconds.
      throughput: (optional) Throughput (in MB/s)
      extras: (optional) Dict mapping string keys to additional benchmark info.
        Values may be either floats or values that are convertible to strings.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the top-level method name.
      metrics: (optional) A list of dict, where each dict has the keys below
        name (required), string, metric name
        value (required), double, metric value
        min_value (optional), double, minimum acceptable metric value
        max_value (optional), double, maximum acceptable metric value
    """
    name = self._get_name(overwrite_name=name)
    _global_report_benchmark(name=name, iters=iters, cpu_time=cpu_time, wall_time=wall_time, throughput=throughput, extras=extras, metrics=metrics)