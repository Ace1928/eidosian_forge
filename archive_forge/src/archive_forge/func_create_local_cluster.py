import collections
from collections import OrderedDict
import contextlib
import functools
import gc
import itertools
import math
import os
import random
import re
import tempfile
import threading
import time
import unittest
from absl.testing import parameterized
import numpy as np
from google.protobuf import descriptor_pool
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.client import session
from tensorflow.python.compat.compat import forward_compatibility_horizon
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _test_metrics_util
from tensorflow.python.framework import config
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import gen_sync_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_ops  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import _pywrap_stacktrace_handler
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import _pywrap_util_port
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.tf_export import tf_export
@tf_export('test.create_local_cluster')
def create_local_cluster(num_workers, num_ps, protocol='grpc', worker_config=None, ps_config=None):
    """Create and start local servers and return the associated `Server` objects.

  "PS" stands for "parameter server": a task responsible for storing and
  updating the model's parameters. Other tasks send updates to these parameters
  as they work on optimizing the parameters. This particular division of labor
  between tasks is not required, but is common for distributed training.

  Read more at https://www.tensorflow.org/guide/extend/architecture

  ![components](https://www.tensorflow.org/images/diag1.svg "components")


  Figure illustrates the interaction of these components.
  "/job:worker/task:0" and "/job:ps/task:0" are both tasks with worker services.


  Example:
  ```python
  workers, _ = tf.test.create_local_cluster(num_workers=2, num_ps=2)

  worker_sessions = [tf.compat.v1.Session(w.target) for w in workers]

  with tf.device("/job:ps/task:0"):
    ...
  with tf.device("/job:ps/task:1"):
    ...
  with tf.device("/job:worker/task:0"):
    ...
  with tf.device("/job:worker/task:1"):
    ...

  worker_sessions[0].run(...)
  ```

  Args:
    num_workers: Number of worker servers to start.
    num_ps: Number of PS servers to start.
    protocol: Communication protocol. Allowed values are documented in the
      documentation of `tf.distribute.Server`.
    worker_config: (optional) `tf.ConfigProto` to initialize workers. Can be
      used to instantiate multiple devices etc.
    ps_config: (optional) `tf.ConfigProto` to initialize PS servers.

  Returns:
    A tuple `(worker_servers, ps_servers)`.  `worker_servers` is a list
    of `num_workers` objects of type `tf.distribute.Server` (all running
    locally);
    and `ps_servers` is a list of `num_ps` objects of similar type.

  Raises:
    ImportError: if portpicker module was not found at load time
  """
    worker_ports = [pick_unused_port() for _ in range(num_workers)]
    ps_ports = [pick_unused_port() for _ in range(num_ps)]
    cluster_dict = {'worker': ['localhost:%s' % port for port in worker_ports], 'ps': ['localhost:%s' % port for port in ps_ports]}
    cs = server_lib.ClusterSpec(cluster_dict)
    workers = [server_lib.Server(cs, job_name='worker', protocol=protocol, task_index=ix, config=worker_config, start=True) for ix in range(num_workers)]
    ps_servers = [server_lib.Server(cs, job_name='ps', protocol=protocol, task_index=ix, config=ps_config, start=True) for ix in range(num_ps)]
    return (workers, ps_servers)