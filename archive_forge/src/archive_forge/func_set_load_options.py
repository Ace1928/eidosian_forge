import contextlib
import threading
def set_load_options(self, load_options):
    self._load_options = load_options
    self._entered_load_context.append(True)