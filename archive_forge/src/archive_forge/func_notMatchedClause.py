from antlr4 import *
from io import StringIO
import sys
def notMatchedClause(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.NotMatchedClauseContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.NotMatchedClauseContext, i)