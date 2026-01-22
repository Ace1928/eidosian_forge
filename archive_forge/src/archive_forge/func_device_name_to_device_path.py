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
def device_name_to_device_path(device_name):
    """Convert device name to device path."""
    device_name_items = compat.as_text(device_name).split('/')
    device_name_items = [item.replace(':', '_') for item in device_name_items]
    return METADATA_FILE_PREFIX + DEVICE_TAG + ','.join(device_name_items)