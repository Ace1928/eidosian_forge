from antlr4 import *
from io import StringIO
import sys
def fuguePartitionNum_sempred(self, localctx: FuguePartitionNumContext, predIndex: int):
    if predIndex == 0:
        return self.precpred(self._ctx, 1)