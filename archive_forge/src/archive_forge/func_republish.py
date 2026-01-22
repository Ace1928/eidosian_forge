import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
def republish(self):
    """Set the timestamp to current and (TODO) update all observers."""
    self.timestamp = time.time()