from zope.interface.declarations import implementer
from twisted.internet.interfaces import (
from twisted.plugin import IPlugin
def parseStreamClient(self, *a, **kw):
    return StreamClient(self, a, kw)