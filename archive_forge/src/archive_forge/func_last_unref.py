import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
@property
def last_unref(self):
    """Last unreference timestamp of this tensor (long integer)."""
    return max(self._unref_times)