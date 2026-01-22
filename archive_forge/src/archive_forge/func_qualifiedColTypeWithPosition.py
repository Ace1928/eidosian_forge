from antlr4 import *
from io import StringIO
import sys
def qualifiedColTypeWithPosition(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.QualifiedColTypeWithPositionContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedColTypeWithPositionContext, i)