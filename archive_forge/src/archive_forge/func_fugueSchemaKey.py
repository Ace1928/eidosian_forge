from antlr4 import *
from io import StringIO
import sys
def fugueSchemaKey(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaKeyContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaKeyContext, i)