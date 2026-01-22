import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mTHURSDAY(self):
    try:
        _type = THURSDAY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('thu')
        alt9 = 2
        LA9_0 = self.input.LA(1)
        if LA9_0 == 114:
            alt9 = 1
        if alt9 == 1:
            pass
            self.match('rsday')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass