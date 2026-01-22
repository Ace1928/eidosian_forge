from antlr4 import *
from io import StringIO
import sys
def inputs2(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(AutolevParser.Inputs2Context)
    else:
        return self.getTypedRuleContext(AutolevParser.Inputs2Context, i)