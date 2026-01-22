import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
@property
def core_metadata(self):
    """Metadata about the `Session.run()` call from the core runtime.

    Of the three counters available in the return value, `global_step` is
    supplied by the caller of the debugged `Session.run()`, while
    `session_run_index` and `executor_step_index` are determined by the state
    of the core runtime, automatically. For the same fetch list, feed keys and
    debug tensor watch options, the same executor will be used and
    `executor_step_index` should increase by one at a time. However, runs with
    different fetch lists, feed keys and debug_tensor watch options that all
    share the same `Session` object can lead to gaps in `session_run_index`.

    Returns:
      If core metadata are loaded, a `namedtuple` with the fields:
        `global_step`: A global step count supplied by the caller of
          `Session.run()`. It is optional to the caller. If the caller did not
          supply this parameter, its value will be -1.
        `session_run_index`: A sorted index for Run() calls to the underlying
          TensorFlow `Session` object.
        `executor_step_index`: A counter for invocations of a given runtime
          executor. The same executor is re-used for the same fetched tensors,
          target nodes, input feed keys and debug tensor watch options.
        `input_names`: Names of the input (feed) Tensors.
        `output_names`: Names of the output (fetched) Tensors.
        `target_nodes`: Names of the target nodes.
      If the core metadata have not been loaded, `None`.
      If more than one core metadata files exist, return a list of the
        `nametuple` described above.
    """
    output = self._core_metadata
    return output[0] if len(output) == 1 else output