import contextlib
import threading
def in_save_context(self):
    return self._in_save_context