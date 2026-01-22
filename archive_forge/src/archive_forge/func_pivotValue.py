from antlr4 import *
from io import StringIO
import sys
def pivotValue(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.PivotValueContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.PivotValueContext, i)