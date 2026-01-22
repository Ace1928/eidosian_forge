import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mJUNE(self):
    try:
        _type = JUNE
        _channel = DEFAULT_CHANNEL
        pass
        self.match('jun')
        alt17 = 2
        LA17_0 = self.input.LA(1)
        if LA17_0 == 101:
            alt17 = 1
        if alt17 == 1:
            pass
            self.match(101)
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass