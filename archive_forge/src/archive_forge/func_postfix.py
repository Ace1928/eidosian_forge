from antlr4 import *
from io import StringIO
import sys
def postfix(self):
    return self.getTypedRuleContext(LaTeXParser.PostfixContext, 0)