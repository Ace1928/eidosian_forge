import contextlib
import threading
def enter_preemption_save_context(self):
    self._in_preemption_save_context = True