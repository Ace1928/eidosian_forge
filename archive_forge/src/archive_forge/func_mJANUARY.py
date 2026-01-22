import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mJANUARY(self):
    try:
        _type = JANUARY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('jan')
        alt13 = 2
        LA13_0 = self.input.LA(1)
        if LA13_0 == 117:
            alt13 = 1
        if alt13 == 1:
            pass
            self.match('uary')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass