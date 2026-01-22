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
@config_proto_serialized.setter
def config_proto_serialized(self, config):
    if isinstance(config, config_pb2.ConfigProto):
        self._config_proto_serialized = config.SerializeToString(deterministic=True)
    elif isinstance(config, str):
        self._config_proto_serialized = config
    elif config is None:
        self._config_proto_serialized = config_pb2.ConfigProto().SerializeToString()
    else:
        raise ValueError('the rewriter config must be either a config_pb2.ConfigProto, or a serialized string of that proto or None. got: {}'.format(type(config)))