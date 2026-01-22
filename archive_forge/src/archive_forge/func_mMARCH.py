import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mMARCH(self):
    try:
        _type = MARCH
        _channel = DEFAULT_CHANNEL
        pass
        self.match('mar')
        alt15 = 2
        LA15_0 = self.input.LA(1)
        if LA15_0 == 99:
            alt15 = 1
        if alt15 == 1:
            pass
            self.match('ch')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass