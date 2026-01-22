import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def do_yes_true(self, state, option):
    assert False, 'do_yes_true can never be entered, but was called with {!r}, {!r}'.format(state, option)