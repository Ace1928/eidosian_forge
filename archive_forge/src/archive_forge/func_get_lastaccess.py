from os import environ
from kivy.logger import Logger
from kivy.clock import Clock
@staticmethod
def get_lastaccess(category, key, default=None):
    """Get the objects last access time in the cache.

        :Parameters:
            `category`: str
                Identifier of the category.
            `key`: str
                Unique identifier of the object in the store.
            `default`: anything, defaults to None
                Default value to be returned if the key is not found.
        """
    try:
        return Cache._objects[category][key]['lastaccess']
    except Exception:
        return default