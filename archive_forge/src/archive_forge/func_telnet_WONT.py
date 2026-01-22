import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def telnet_WONT(self, option):
    s = self.getOptionState(option)
    self.wontMap[s.him.state, s.him.negotiating](self, s, option)