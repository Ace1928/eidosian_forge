import sys
from antlr3 import *
from antlr3.compat import set, frozenset
def synpred2_Groc(self):
    self._state.backtracking += 1
    start = self.input.mark()
    try:
        self.synpred2_Groc_fragment()
    except BacktrackingFailed:
        success = False
    else:
        success = True
    self.input.rewind(start)
    self._state.backtracking -= 1
    return success