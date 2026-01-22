from antlr4 import *
from io import StringIO
import sys
def fuguePartitionNum(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FuguePartitionNumContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionNumContext, i)