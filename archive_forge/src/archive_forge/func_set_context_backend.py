import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def set_context_backend(type):
    """
    Sets the internal context backend used to track execution context.

    type must be one of 'greenlet' or 'native_thread'. For example:

    >>> import greenlet, yappi
    >>> yappi.set_context_backend("greenlet")

    Setting the context backend will reset any callbacks configured via:
      - set_context_id_callback
      - set_context_name_callback

    The default callbacks for the backend provided will be installed instead.
    Configure the callbacks each time after setting context backend.
    """
    type = type.upper()
    if type not in BACKEND_TYPES:
        raise YappiError(f'Invalid backend type: {type}')
    if type == GREENLET:
        id_cbk, name_cbk = _create_greenlet_callbacks()
        _yappi.set_context_id_callback(id_cbk)
        set_context_name_callback(name_cbk)
    else:
        _yappi.set_context_id_callback(None)
        set_context_name_callback(None)
    _yappi.set_context_backend(BACKEND_TYPES[type])