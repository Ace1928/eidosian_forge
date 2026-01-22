import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def period(self):
    retval = self.period_return()
    retval.start = self.input.LT(1)
    try:
        try:
            pass
            if HOURS <= self.input.LA(1) <= MINUTES:
                self.input.consume()
                self._state.errorRecovery = False
            else:
                mse = MismatchedSetException(None, self.input)
                raise mse
            retval.stop = self.input.LT(-1)
        except RecognitionException as re:
            self.reportError(re)
            self.recover(self.input, re)
    finally:
        pass
    return retval