import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mDECEMBER(self):
    try:
        _type = DECEMBER
        _channel = DEFAULT_CHANNEL
        pass
        self.match('dec')
        alt23 = 2
        LA23_0 = self.input.LA(1)
        if LA23_0 == 101:
            alt23 = 1
        if alt23 == 1:
            pass
            self.match('ember')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass