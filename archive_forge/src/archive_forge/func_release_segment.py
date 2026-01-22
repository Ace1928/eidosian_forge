import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def release_segment(self, c, segment_name):
    """Calls unlink() on the shared memory block with the supplied name
            and removes it from the tracker instance inside the Server."""
    self.shared_memory_context.destroy_segment(segment_name)