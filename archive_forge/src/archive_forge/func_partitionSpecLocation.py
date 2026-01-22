from antlr4 import *
from io import StringIO
import sys
def partitionSpecLocation(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.PartitionSpecLocationContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecLocationContext, i)