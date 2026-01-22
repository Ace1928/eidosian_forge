from twisted.python import roots
from twisted.web import pages, resource
def removeHost(self, name):
    """Remove a host."""
    del self.hosts[name]