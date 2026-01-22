from antlr4 import *
from io import StringIO
import sys
def valueExpression(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)