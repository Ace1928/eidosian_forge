from antlr4 import *
from io import StringIO
import sys
def mp_nofunc_sempred(self, localctx: Mp_nofuncContext, predIndex: int):
    if predIndex == 3:
        return self.precpred(self._ctx, 2)