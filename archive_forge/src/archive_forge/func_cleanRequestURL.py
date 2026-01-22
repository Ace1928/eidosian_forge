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
def cleanRequestURL(url):
    """
    Clean a URL from a Request line.
    """
    url.transport = None
    url.maddr = None
    url.ttl = None
    url.headers = {}