import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
def end_event(kind, data=None, exc_details=None):
    """Trigger the end of an event of *kind*, *exc_details*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``. Or, ``None`` if no error.
    """
    evt = Event(kind=kind, status=EventStatus.END, data=data, exc_details=exc_details)
    broadcast(evt)