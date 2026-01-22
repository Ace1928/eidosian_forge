import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def mTWO_DIGIT_HOUR_TIME(self):
    try:
        _type = TWO_DIGIT_HOUR_TIME
        _channel = DEFAULT_CHANNEL
        pass
        pass
        alt1 = 3
        LA1 = self.input.LA(1)
        if LA1 == 48:
            alt1 = 1
        elif LA1 == 49:
            alt1 = 2
        elif LA1 == 50:
            alt1 = 3
        else:
            if self._state.backtracking > 0:
                raise BacktrackingFailed
            nvae = NoViableAltException('', 1, 0, self.input)
            raise nvae
        if alt1 == 1:
            pass
            pass
            self.match(48)
            self.mDIGIT()
        elif alt1 == 2:
            pass
            pass
            self.match(49)
            self.mDIGIT()
        elif alt1 == 3:
            pass
            pass
            self.match(50)
            self.matchRange(48, 51)
        self.match(58)
        pass
        self.matchRange(48, 53)
        self.mDIGIT()
        if self._state.backtracking == 0:
            _type = TIME
        self._state.type = _type
        self._state.channel = _channel
    finally:
        pass