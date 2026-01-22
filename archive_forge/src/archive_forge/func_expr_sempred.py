from antlr4 import *
from io import StringIO
import sys
def expr_sempred(self, localctx: ExprContext, predIndex: int):
    if predIndex == 0:
        return self.precpred(self._ctx, 16)
    if predIndex == 1:
        return self.precpred(self._ctx, 15)
    if predIndex == 2:
        return self.precpred(self._ctx, 14)
    if predIndex == 3:
        return self.precpred(self._ctx, 3)
    if predIndex == 4:
        return self.precpred(self._ctx, 2)