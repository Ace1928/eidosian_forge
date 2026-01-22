from antlr4 import *
from io import StringIO
import sys
def fugueWildSchemaPair(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FugueWildSchemaPairContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FugueWildSchemaPairContext, i)