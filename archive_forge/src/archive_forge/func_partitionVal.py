from antlr4 import *
from io import StringIO
import sys
def partitionVal(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.PartitionValContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.PartitionValContext, i)