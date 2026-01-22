from antlr4 import *
from io import StringIO
import sys
def relation(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(LaTeXParser.RelationContext)
    else:
        return self.getTypedRuleContext(LaTeXParser.RelationContext, i)