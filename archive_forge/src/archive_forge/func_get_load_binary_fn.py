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
def get_load_binary_fn(self):
    """
        Return a callable to load binary
        """
    raise NotImplementedError