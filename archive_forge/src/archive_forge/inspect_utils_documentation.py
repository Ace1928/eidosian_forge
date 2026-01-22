import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
Detects what future imports are necessary to safely execute entity source.

  Args:
    entity: Any object

  Returns:
    A tuple of future strings
  