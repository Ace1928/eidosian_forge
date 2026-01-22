from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def hasEmptyPath(self):
    return self.getReturnState(len(self) - 1) == self.EMPTY_RETURN_STATE