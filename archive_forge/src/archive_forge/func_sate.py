import copy
import sys
import textwrap
import traceback
import types
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
def sate(self):
    self._sated = True
    self._type = None
    self._repr = None
    self._stack_frame = None
    self._logging_module = None