from antlr4 import *
from io import StringIO
import sys
def identifierComment(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.IdentifierCommentContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierCommentContext, i)