import sys
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols import basic
def sendUIDLResp(self, req):
    self.sendLine(UIDL)