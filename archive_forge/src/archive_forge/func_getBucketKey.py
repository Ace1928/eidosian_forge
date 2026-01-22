from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp
def getBucketKey(self, transport):
    return transport.getHost()[2]