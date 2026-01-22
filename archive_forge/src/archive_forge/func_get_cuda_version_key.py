import functools
import hashlib
import importlib
import importlib.util
import os
import re
import subprocess
import traceback
from typing import Dict
from ..runtime.driver import DriverBase
def get_cuda_version_key():
    global _cached_cuda_version_key
    if _cached_cuda_version_key is None:
        key = compute_core_version_key()
        try:
            ptxas = path_to_ptxas()[0]
            ptxas_version = subprocess.check_output([ptxas, '--version'])
        except RuntimeError:
            ptxas_version = b'NO_PTXAS'
        _cached_cuda_version_key = key + '-' + hashlib.sha1(ptxas_version).hexdigest()
    return _cached_cuda_version_key