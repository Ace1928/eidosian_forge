from antlr4 import *
from io import StringIO
import sys
def fromStatement(self):
    return self.getTypedRuleContext(fugue_sqlParser.FromStatementContext, 0)