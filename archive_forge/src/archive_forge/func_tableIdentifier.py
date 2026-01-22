from antlr4 import *
from io import StringIO
import sys
def tableIdentifier(self):
    return self.getTypedRuleContext(fugue_sqlParser.TableIdentifierContext, 0)