from antlr4 import *
from io import StringIO
import sys
def singleDataType(self):
    localctx = fugue_sqlParser.SingleDataTypeContext(self, self._ctx, self.state)
    self.enterRule(localctx, 162, self.RULE_singleDataType)
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 1166
        self.dataType()
        self.state = 1167
        self.match(fugue_sqlParser.EOF)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx