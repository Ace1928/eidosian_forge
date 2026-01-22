import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def weekdays(self):
    try:
        try:
            pass
            alt10 = 2
            LA10_0 = self.input.LA(1)
            if LA10_0 == DAY:
                alt10 = 1
            elif MONDAY <= LA10_0 <= SUNDAY:
                alt10 = 2
            else:
                nvae = NoViableAltException('', 10, 0, self.input)
                raise nvae
            if alt10 == 1:
                pass
                self.match(self.input, DAY, self.FOLLOW_DAY_in_weekdays365)
                if self.ordinal_set:
                    self.monthday_set = self.ordinal_set
                    self.ordinal_set = set()
                else:
                    self.ordinal_set = self.ordinal_set.union(allOrdinals)
                    self.weekday_set = set([self.ValueOf(SUNDAY), self.ValueOf(MONDAY), self.ValueOf(TUESDAY), self.ValueOf(WEDNESDAY), self.ValueOf(THURSDAY), self.ValueOf(FRIDAY), self.ValueOf(SATURDAY), self.ValueOf(SUNDAY)])
            elif alt10 == 2:
                pass
                pass
                self._state.following.append(self.FOLLOW_weekday_in_weekdays373)
                self.weekday()
                self._state.following.pop()
                while True:
                    alt9 = 2
                    LA9_0 = self.input.LA(1)
                    if LA9_0 == COMMA:
                        alt9 = 1
                    if alt9 == 1:
                        pass
                        self.match(self.input, COMMA, self.FOLLOW_COMMA_in_weekdays376)
                        self._state.following.append(self.FOLLOW_weekday_in_weekdays378)
                        self.weekday()
                        self._state.following.pop()
                    else:
                        break
                if not self.ordinal_set:
                    self.ordinal_set = self.ordinal_set.union(allOrdinals)
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return