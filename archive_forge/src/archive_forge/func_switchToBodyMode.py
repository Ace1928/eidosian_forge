import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
def switchToBodyMode(self, decoder):
    """
        Switch to body parsing mode - interpret any more bytes delivered as
        part of the message body and deliver them to the given decoder.
        """
    if self.state == BODY:
        raise RuntimeError('already in body mode')
    self.bodyDecoder = decoder
    self.state = BODY
    self.setRawMode()