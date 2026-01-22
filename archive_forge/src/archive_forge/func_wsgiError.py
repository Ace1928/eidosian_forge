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
def wsgiError(started, type, value, traceback):
    self._log.failure('WSGI application error', failure=Failure(value, type, traceback))
    if started:
        self.request.loseConnection()
    else:
        self.request.setResponseCode(INTERNAL_SERVER_ERROR)
        self.request.finish()