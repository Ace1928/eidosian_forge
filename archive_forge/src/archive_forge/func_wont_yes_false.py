import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def wont_yes_false(self, state, option):
    state.him.state = 'no'
    self.disableRemote(option)
    self._dont(option)