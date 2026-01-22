from antlr4 import *
from io import StringIO
import sys
def varType(self):
    return self.getTypedRuleContext(AutolevParser.VarTypeContext, 0)