import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def call_me_and_unregister():
    signals.unregister_on_hangup('myid')
    calls.append('called_and_unregistered')