import sys
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols import basic
def sendCapabilities(self):
    if self.caps is None:
        self.caps = [CAP_START]
    if UIDL_SUPPORT:
        self.caps.append(CAPABILITIES_UIDL)
    if SSL_SUPPORT:
        self.caps.append(CAPABILITIES_SSL)
    for cap in CAPABILITIES:
        self.caps.append(cap)
    resp = b'\r\n'.join(self.caps)
    resp += b'\r\n.'
    self.sendLine(resp)