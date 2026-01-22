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
def register_backend(device_type: str, backend_cls: type):
    if device_type not in _backends:
        _backends[device_type] = backend_cls.create_backend(device_type)