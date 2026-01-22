from antlr4 import *
from io import StringIO
import sys
def strictNonReserved(self):
    return self.getTypedRuleContext(fugue_sqlParser.StrictNonReservedContext, 0)