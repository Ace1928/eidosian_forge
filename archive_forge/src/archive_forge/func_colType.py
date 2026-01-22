from antlr4 import *
from io import StringIO
import sys
def colType(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.ColTypeContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.ColTypeContext, i)