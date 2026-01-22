from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorWin32Events
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID, isInIOThread
def occurred(self):
    self.success = True
    self.eventThreadID = getThreadID()
    self._finished.callback(None)