from queue import Empty, Queue
from twisted.internet import _threadedselect
from twisted.python import log, runtime
def registerWxApp(self, wxapp):
    """
        Register wxApp instance with the reactor.
        """
    self.wxapp = wxapp