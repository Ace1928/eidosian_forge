import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
@property
def is_end(self):
    """Is it an *END* event?

        Returns
        -------
        res : bool
        """
    return self._status == EventStatus.END