import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mTUESDAY(self):
    try:
        _type = TUESDAY
        _channel = DEFAULT_CHANNEL
        pass
        self.match('tue')
        alt7 = 2
        LA7_0 = self.input.LA(1)
        if LA7_0 == 115:
            alt7 = 1
        if alt7 == 1:
            pass
            self.match('sday')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass