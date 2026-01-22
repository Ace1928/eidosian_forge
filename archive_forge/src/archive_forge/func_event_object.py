import atexit
import queue
import threading
import weakref
@staticmethod
def event_object(*args, **kwargs):
    return threading.Event(*args, **kwargs)