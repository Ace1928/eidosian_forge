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
def extended_timestamp(self):
    """Extended timestamp, possibly with an index suffix.

    The index suffix, e.g., "-1", is for disambiguating multiple dumps of the
    same tensor with the same timestamp, which can occur if the dumping events
    are spaced by shorter than the temporal resolution of the timestamps.

    Returns:
      (`str`) The extended timestamp.
    """
    return self._extended_timestamp