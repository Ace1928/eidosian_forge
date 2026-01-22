from antlr4 import *
from io import StringIO
import sys
def fugueLoadTask(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueLoadTaskContext, 0)