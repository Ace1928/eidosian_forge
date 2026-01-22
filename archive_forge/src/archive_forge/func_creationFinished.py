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
def creationFinished(self):
    if self.length != None and self.length != len(self.body):
        raise ValueError('wrong body length')
    self.finished = 1