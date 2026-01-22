from antlr4 import *
from io import StringIO
import sys
def groupingSet(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.GroupingSetContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.GroupingSetContext, i)