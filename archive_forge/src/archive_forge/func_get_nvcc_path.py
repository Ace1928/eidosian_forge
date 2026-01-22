import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def get_nvcc_path():
    global _nvcc_path
    if _nvcc_path == '':
        _nvcc_path = _get_nvcc_path()
    return _nvcc_path