import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def get_k8s_kubeconfig(env_var: str='KUBECONFIG', default: str=None) -> str:
    """
    Get the kubernetes kubeconfig.
    """
    global _k8s_kubeconfig
    if _k8s_kubeconfig is None:
        if is_in_kubernetes():
            _k8s_kubeconfig = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        else:
            _k8s_kubeconfig = os.getenv(env_var, default)
            if _k8s_kubeconfig is None:
                _k8s_kubeconfig = os.path.expanduser('~/.kube/config')
    return _k8s_kubeconfig