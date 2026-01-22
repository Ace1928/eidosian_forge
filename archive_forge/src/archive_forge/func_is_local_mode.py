import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
def is_local_mode() -> bool:
    """Returns true if DTensor shall run in local mode."""
    return not jobs()