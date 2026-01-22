import contextlib
import threading
def enter_save_context(self, options):
    self._in_save_context = True
    self._options = options