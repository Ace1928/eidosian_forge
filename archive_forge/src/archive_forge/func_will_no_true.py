import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def will_no_true(self, state, option):
    state.him.state = 'yes'
    state.him.negotiating = False
    d = state.him.onResult
    state.him.onResult = None
    d.callback(True)
    assert self.enableRemote(option), 'enableRemote must return True in this context (for option {!r})'.format(option)