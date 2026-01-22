from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
def unregisterAsAccountClient(self):
    """Tell the chat UI that I have `signed off'."""
    self.chat.unregisterAccountClient(self)