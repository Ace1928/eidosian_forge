from weakref import ref
from contextlib import contextmanager
from threading import current_thread, RLock
def create_dict(self):
    """Create a new dict for the current thread, and return it."""
    localdict = {}
    key = self.key
    thread = current_thread()
    idt = id(thread)

    def local_deleted(_, key=key):
        thread = wrthread()
        if thread is not None:
            del thread.__dict__[key]

    def thread_deleted(_, idt=idt):
        local = wrlocal()
        if local is not None:
            dct = local.dicts.pop(idt)
    wrlocal = ref(self, local_deleted)
    wrthread = ref(thread, thread_deleted)
    thread.__dict__[key] = wrlocal
    self.dicts[idt] = (wrthread, localdict)
    return localdict