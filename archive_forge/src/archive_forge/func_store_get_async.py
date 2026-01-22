from kivy.clock import Clock
from kivy.event import EventDispatcher
def store_get_async(self, key, callback):
    try:
        value = self.store_get(key)
        callback(self, key, value)
    except KeyError:
        callback(self, key, None)