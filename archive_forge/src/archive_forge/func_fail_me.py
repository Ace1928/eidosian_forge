import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def fail_me():
    raise RuntimeError('something bad happened')