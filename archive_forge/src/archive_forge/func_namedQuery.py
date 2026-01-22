from antlr4 import *
from io import StringIO
import sys
def namedQuery(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.NamedQueryContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.NamedQueryContext, i)