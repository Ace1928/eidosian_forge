from antlr4 import *
from io import StringIO
import sys
def postfix_nofunc(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(LaTeXParser.Postfix_nofuncContext)
    else:
        return self.getTypedRuleContext(LaTeXParser.Postfix_nofuncContext, i)