import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def ordinals(self):
    try:
        try:
            pass
            alt7 = 2
            LA7_0 = self.input.LA(1)
            if LA7_0 == EVERY:
                alt7 = 1
            elif FIRST <= LA7_0 <= FOURTH_OR_FIFTH:
                alt7 = 2
            else:
                nvae = NoViableAltException('', 7, 0, self.input)
                raise nvae
            if alt7 == 1:
                pass
                self.match(self.input, EVERY, self.FOLLOW_EVERY_in_ordinals218)
            elif alt7 == 2:
                pass
                pass
                self._state.following.append(self.FOLLOW_ordinal_in_ordinals226)
                self.ordinal()
                self._state.following.pop()
                while True:
                    alt6 = 2
                    LA6_0 = self.input.LA(1)
                    if LA6_0 == COMMA:
                        alt6 = 1
                    if alt6 == 1:
                        pass
                        self.match(self.input, COMMA, self.FOLLOW_COMMA_in_ordinals229)
                        self._state.following.append(self.FOLLOW_ordinal_in_ordinals231)
                        self.ordinal()
                        self._state.following.pop()
                    else:
                        break
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return