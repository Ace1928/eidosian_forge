from collections import defaultdict, namedtuple
import gc
import os
import re
import time
import tracemalloc
from typing import Callable, List, Optional
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def disable_log_once_globally():
    """Make log_once() return False in this process."""
    global _disabled
    _disabled = True