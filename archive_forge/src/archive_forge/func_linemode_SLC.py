import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def linemode_SLC(self, data):
    chunks = zip(*[iter(data)] * 3)
    for slcFunction, slcValue, slcWhat in chunks:
        ('SLC', ord(slcFunction), ord(slcValue), ord(slcWhat))