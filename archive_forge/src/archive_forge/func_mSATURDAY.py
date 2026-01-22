import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mSATURDAY(self):
    try:
        _type = SATURDAY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('sat')
        alt11 = 2
        LA11_0 = self.input.LA(1)
        if LA11_0 == 117:
            alt11 = 1
        if alt11 == 1:
            pass
            self.match('urday')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass