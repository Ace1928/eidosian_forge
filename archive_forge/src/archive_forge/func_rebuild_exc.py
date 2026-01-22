import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
def rebuild_exc(exc, tb):
    exc.__cause__ = RemoteTraceback(tb)
    return exc