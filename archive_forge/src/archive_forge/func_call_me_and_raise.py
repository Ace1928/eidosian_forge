import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def call_me_and_raise():
    raise KeyboardInterrupt()