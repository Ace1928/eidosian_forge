import os
import socket
import sys
import threading
import traceback
from contextlib import contextmanager
from threading import TIMEOUT_MAX as THREAD_TIMEOUT_MAX
from celery.local import Proxy
def on_crash(self, msg, *fmt, **kwargs):
    print(msg.format(*fmt), file=sys.stderr)
    traceback.print_exc(None, sys.stderr)