from antlr4 import *
from io import StringIO
import sys
def fugueColSort(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FugueColSortContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FugueColSortContext, i)