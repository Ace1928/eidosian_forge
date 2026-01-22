import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.num_clients', v1=[])
def num_clients() -> int:
    """Returns the number of clients in this DTensor cluster."""
    if is_local_mode():
        return 1
    return len(jobs())