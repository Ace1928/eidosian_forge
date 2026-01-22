from antlr4 import *
from io import StringIO
import sys
def transformArgument(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.TransformArgumentContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.TransformArgumentContext, i)