import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.heartbeat_enabled', v1=[])
def heartbeat_enabled() -> bool:
    """Returns true if DTensor heartbeat service is enabled."""
    return os.environ.get(_DT_HEARTBEAT_ENABLED, 'true').lower() in ('true', '1')