import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def is_flag_on(self, flag_name):
    """Returns True if the given flag is on."""
    found, flag_value = self.get_flag_value(flag_name)
    if not found:
        return False
    if flag_value is None:
        return True
    flag_value = flag_value.lower()
    enabled = flag_value in ['1', 't', 'true', 'y', 'yes']
    return enabled