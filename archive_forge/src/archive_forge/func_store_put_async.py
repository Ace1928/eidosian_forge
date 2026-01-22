from kivy.clock import Clock
from kivy.event import EventDispatcher
def store_put_async(self, key, value, callback):
    try:
        value = self.put(key, **value)
        callback(self, key, value)
    except:
        callback(self, key, None)