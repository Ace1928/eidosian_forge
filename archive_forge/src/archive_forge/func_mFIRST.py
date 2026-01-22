import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mFIRST(self):
    try:
        _type = FIRST
        _channel = DEFAULT_CHANNEL
        pass
        alt2 = 2
        LA2_0 = self.input.LA(1)
        if LA2_0 == 49:
            alt2 = 1
        elif LA2_0 == 102:
            alt2 = 2
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed
            nvae = NoViableAltException('', 2, 0, self.input)
            raise nvae
        if alt2 == 1:
            pass
            self.match('1st')
        elif alt2 == 2:
            pass
            self.match('first')
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass