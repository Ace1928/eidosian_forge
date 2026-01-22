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
def extract_core_metadata_from_event_proto(event):
    json_metadata = json.loads(event.log_message.message)
    return _CoreMetadata(json_metadata['global_step'], json_metadata['session_run_index'], json_metadata['executor_step_index'], json_metadata['input_names'], json_metadata['output_names'], json_metadata['target_nodes'])