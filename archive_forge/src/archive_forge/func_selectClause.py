from antlr4 import *
from io import StringIO
import sys
def selectClause(self):
    return self.getTypedRuleContext(fugue_sqlParser.SelectClauseContext, 0)