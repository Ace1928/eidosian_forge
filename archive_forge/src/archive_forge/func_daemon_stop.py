import functools
import logging
from multiprocessing import managers
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from oslo_rootwrap import cmd
from oslo_rootwrap import jsonrpc
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def daemon_stop(server, signal, frame):
    LOG.info('Got signal %s. Shutting down server', signal)
    try:
        server.stop_event.set()
    except AttributeError:
        raise KeyboardInterrupt