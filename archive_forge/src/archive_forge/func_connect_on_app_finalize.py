import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
def connect_on_app_finalize(callback):
    """Connect callback to be called when any app is finalized."""
    _on_app_finalizers.add(callback)
    return callback