import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mJULY(self):
    try:
        _type = JULY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('jul')
        alt18 = 2
        LA18_0 = self.input.LA(1)
        if LA18_0 == 121:
            alt18 = 1
        if alt18 == 1:
            pass
            self.match(121)
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass