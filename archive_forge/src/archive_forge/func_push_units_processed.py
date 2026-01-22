import time
def push_units_processed(self, n):
    self._units_processed.append(n)
    if len(self._units_processed) > self._window_size:
        self._units_processed.pop(0)