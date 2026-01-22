import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def get_hipcc_path():
    global _hipcc_path
    if _hipcc_path == '':
        _hipcc_path = _get_hipcc_path()
    return _hipcc_path