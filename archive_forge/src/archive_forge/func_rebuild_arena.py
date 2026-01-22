import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def rebuild_arena(size, dupfd):
    return Arena(size, dupfd.detach())