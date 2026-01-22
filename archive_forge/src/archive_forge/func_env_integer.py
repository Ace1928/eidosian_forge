import os
import sys
from ray._private.ray_constants import (  # noqa F401
def env_integer(key, default):
    if key in os.environ:
        val = os.environ[key]
        if val == 'inf':
            return sys.maxsize
        else:
            return int(val)
    return default