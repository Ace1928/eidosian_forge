import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def get_local_kubeconfig(name: Optional[str]=None, set_as_envval: bool=False) -> str:
    """
    Get the local kubernetes kubeconfig
    - this assumes that it's not the default one.
    """
    if name is None:
        return get_k8s_kubeconfig()
    p = os.path.join(get_local_kubeconfig_dir(), name)
    for stem in ('', '-cluster', '-admin-config', '-config', '-kubeconfig'):
        for ext in ('', '.yaml', '.yml'):
            if os.path.isfile(p + stem + ext):
                px = os.path.abspath(p + stem + ext)
                if set_as_envval:
                    os.environ['KUBECONFIG'] = px
                return px
    raise FileNotFoundError(f'Could not find kubeconfig file: {name} @ {p}')