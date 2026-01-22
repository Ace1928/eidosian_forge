import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def dont_no_true(self, state, option):
    state.us.negotiating = False
    d = state.us.onResult
    state.us.onResult = None
    d.errback(OptionRefused(option))