from antlr4 import *
from io import StringIO
import sys
def queryTerm(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.QueryTermContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.QueryTermContext, i)