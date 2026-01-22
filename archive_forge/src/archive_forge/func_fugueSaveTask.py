from antlr4 import *
from io import StringIO
import sys
def fugueSaveTask(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueSaveTaskContext, 0)