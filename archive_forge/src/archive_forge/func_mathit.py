from antlr4 import *
from io import StringIO
import sys
def mathit(self):
    return self.getTypedRuleContext(LaTeXParser.MathitContext, 0)