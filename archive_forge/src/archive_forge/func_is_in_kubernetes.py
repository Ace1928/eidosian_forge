import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def is_in_kubernetes() -> bool:
    """
    Check if we are running in a kubernetes cluster.
    """
    global _is_in_kubernetes
    if _is_in_kubernetes is None:
        _is_in_kubernetes = os.path.isdir('/var/run/secrets/kubernetes.io/')
    return _is_in_kubernetes