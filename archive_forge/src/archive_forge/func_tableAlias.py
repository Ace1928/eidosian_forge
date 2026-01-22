from antlr4 import *
from io import StringIO
import sys
def tableAlias(self):
    return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)