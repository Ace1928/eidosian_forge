from kivy.clock import Clock
from kivy.event import EventDispatcher
def async_count(self, callback):
    """Asynchronously return the number of entries in the storage.
        """
    self._schedule(self.store_count_async, callback=callback)