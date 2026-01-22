from antlr4 import *
from io import StringIO
import sys
def intervalUnit(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.IntervalUnitContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.IntervalUnitContext, i)