import os
import threading
import time
from typing import Any, List, Optional, Text
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.summary_io import SummaryWriterCache
def end_of_blocking_time():
    blocking_end_time = time.time()
    metrics.AddCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT_V1, microseconds=_get_duration_microseconds(blocking_start_time, blocking_end_time))