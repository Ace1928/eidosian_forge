from antlr4 import *
from io import StringIO
import sys
def locationSpec(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)