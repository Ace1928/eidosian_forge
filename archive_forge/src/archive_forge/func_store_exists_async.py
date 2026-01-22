from kivy.clock import Clock
from kivy.event import EventDispatcher
def store_exists_async(self, key, callback):
    try:
        value = self.store_exists(key)
        callback(self, key, value)
    except:
        callback(self, key, None)