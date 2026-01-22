from collections import defaultdict, namedtuple
import gc
import os
import re
import time
import tracemalloc
from typing import Callable, List, Optional
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def enable_periodic_logging():
    """Make log_once() periodically return True in this process."""
    global _periodic_log
    _periodic_log = True