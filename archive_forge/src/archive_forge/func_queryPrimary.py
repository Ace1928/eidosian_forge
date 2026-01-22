from antlr4 import *
from io import StringIO
import sys
def queryPrimary(self):
    return self.getTypedRuleContext(fugue_sqlParser.QueryPrimaryContext, 0)