import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def get_k8s_namespace(env_var: str='NAMESPACE', default: str='default') -> str:
    """
    Get the kubernetes namespace.
    """
    global _k8s_namespace
    if _k8s_namespace is None:
        if is_in_kubernetes():
            with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace') as f:
                _k8s_namespace = f.read().strip()
        else:
            _k8s_namespace = os.getenv(env_var, default)
    return _k8s_namespace