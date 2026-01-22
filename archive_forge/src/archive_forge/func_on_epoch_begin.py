import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver
def on_epoch_begin(self, epoch, logs=None):
    """Add histogram op to Model eval_function callbacks, reset batch count."""
    if self.histogram_freq and epoch % self.histogram_freq == 0:
        self.model._make_test_function()
        if self.merged not in self.model.test_function.fetches:
            self.model.test_function.fetches.append(self.merged)
            self.model.test_function.fetch_callbacks[self.merged] = self._fetch_callback