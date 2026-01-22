from threading import local
from typing import Dict, Type
def currentContext(self):
    try:
        return self.storage.ct
    except AttributeError:
        ct = self.storage.ct = ContextTracker()
        return ct