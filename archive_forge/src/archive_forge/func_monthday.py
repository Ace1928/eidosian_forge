import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def monthday(self):
    day = None
    try:
        try:
            pass
            day = self.input.LT(1)
            if DIGIT <= self.input.LA(1) <= DIGITS:
                self.input.consume()
                self._state.errorRecovery = False
            else:
                mse = MismatchedSetException(None, self.input)
                raise mse
            self.monthday_set.add(int(day.text))
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return