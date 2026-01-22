from antlr4 import *
from io import StringIO
import sys
def postfix_op(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(LaTeXParser.Postfix_opContext)
    else:
        return self.getTypedRuleContext(LaTeXParser.Postfix_opContext, i)