import contextlib
import threading
def exit_async_metrics_context(self):
    self._in_async_metrics_context = False