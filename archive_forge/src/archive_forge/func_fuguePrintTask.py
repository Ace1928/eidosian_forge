from antlr4 import *
from io import StringIO
import sys
def fuguePrintTask(self):
    return self.getTypedRuleContext(fugue_sqlParser.FuguePrintTaskContext, 0)