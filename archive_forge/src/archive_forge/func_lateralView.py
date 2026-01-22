from antlr4 import *
from io import StringIO
import sys
def lateralView(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.LateralViewContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.LateralViewContext, i)