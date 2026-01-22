from antlr4 import *
from io import StringIO
import sys
def fromClause(self):
    return self.getTypedRuleContext(fugue_sqlParser.FromClauseContext, 0)