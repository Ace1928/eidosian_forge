from collections.abc import Sequence
from sys import exc_info
from warnings import warn
from zope.interface import implementer
from twisted.internet.threads import blockingCallFromThread
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.web.http import INTERNAL_SERVER_ERROR
from twisted.web.resource import IResource
from twisted.web.server import NOT_DONE_YET
def wsgiWrite(started):
    if not started:
        self._sendResponseHeaders()
    self.request.write(data)