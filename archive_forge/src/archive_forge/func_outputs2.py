from antlr4 import *
from io import StringIO
import sys
def outputs2(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(AutolevParser.Outputs2Context)
    else:
        return self.getTypedRuleContext(AutolevParser.Outputs2Context, i)