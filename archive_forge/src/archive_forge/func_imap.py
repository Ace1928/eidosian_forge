import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
def imap(self, func: Callable, iterable: Iterable, chunksize: Optional[int]=1):
    """Same as `map`, but only submits one batch of tasks to each actor
        process at a time.

        This can be useful if the iterable of arguments is very large or each
        task's arguments consumes a large amount of resources.

        The results are returned in the order corresponding to their arguments
        in the iterable.

        Returns:
            OrderedIMapIterator
        """
    self._check_running()
    return OrderedIMapIterator(self, func, iterable, chunksize=chunksize)