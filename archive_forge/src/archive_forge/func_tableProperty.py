from antlr4 import *
from io import StringIO
import sys
def tableProperty(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.TablePropertyContext, i)