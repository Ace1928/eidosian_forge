import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
Creates a new MonitoredTimer.

    Args:
      cell: the cell associated with the time metric that will be inremented.
      monitored_section_name: name of action being monitored here.
      avoid_repetitive_counting: when set to True, if already in a monitored
        timer section with the same monitored_section_name, skip counting.
    