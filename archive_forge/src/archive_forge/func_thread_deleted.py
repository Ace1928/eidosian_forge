from weakref import ref
from contextlib import contextmanager
from threading import current_thread, RLock
def thread_deleted(_, idt=idt):
    local = wrlocal()
    if local is not None:
        dct = local.dicts.pop(idt)