from kivy.clock import Clock
from kivy.event import EventDispatcher
def store_find_async(self, filters, callback):
    for key, entry in self.store_find(filters):
        callback(self, filters, key, entry)
    callback(self, filters, None, None)