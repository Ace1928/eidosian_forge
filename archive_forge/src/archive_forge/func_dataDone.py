import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
def dataDone(self):
    """
        Clear out any buffered data that may be hanging around.
        """
    self.clearLineBuffer()
    if self.state == 'firstline':
        return
    if self.state != 'body':
        self.reset()
        return
    if self.length == None:
        self.messageDone()
    elif self.length < self.bodyReceived:
        self.reset()
    else:
        raise RuntimeError('this should never happen')