from antlr4 import *
from io import StringIO
import sys
def frameBound(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FrameBoundContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FrameBoundContext, i)