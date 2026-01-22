from antlr4 import *
from io import StringIO
import sys
def hintStatement(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.HintStatementContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.HintStatementContext, i)