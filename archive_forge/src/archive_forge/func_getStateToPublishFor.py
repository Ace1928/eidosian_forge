import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
def getStateToPublishFor(self, perspective):
    """Implement me to special-case your state for a perspective."""
    return self.getStateToPublish()