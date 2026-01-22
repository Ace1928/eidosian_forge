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
def convert_to_error(kind, result):
    if kind == '#ERROR':
        return result
    elif kind in ('#TRACEBACK', '#UNSERIALIZABLE'):
        if not isinstance(result, str):
            raise TypeError("Result {0!r} (kind '{1}') type is {2}, not str".format(result, kind, type(result)))
        if kind == '#UNSERIALIZABLE':
            return RemoteError('Unserializable message: %s\n' % result)
        else:
            return RemoteError(result)
    else:
        return ValueError('Unrecognized message type {!r}'.format(kind))