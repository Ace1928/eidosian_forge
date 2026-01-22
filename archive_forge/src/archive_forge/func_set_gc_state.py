import weakref
import gc
from contextlib import contextmanager
from platform import python_implementation
def set_gc_state(state):
    """ Set status of garbage collector """
    if gc.isenabled() == state:
        return
    if state:
        gc.enable()
    else:
        gc.disable()