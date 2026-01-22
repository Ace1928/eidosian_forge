import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
def is_gpu_present() -> bool:
    """Returns true if TPU devices are present."""
    return bool(tf_config.list_physical_devices('GPU'))