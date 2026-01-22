import sys
import os
import threading
import collections
import time
import types
import weakref
import errno
from queue import Empty, Full
from . import connection
from . import context
from .util import debug, info, Finalize, register_after_fork, is_exiting

        Private API hook called when feeding data in the background thread
        raises an exception.  For overriding by concurrent.futures.
        