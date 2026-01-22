from antlr4 import *
from io import StringIO
import sys
def singleTableSchema(self):
    localctx = fugue_sqlParser.SingleTableSchemaContext(self, self._ctx, self.state)
    self.enterRule(localctx, 164, self.RULE_singleTableSchema)
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 1169
        self.colTypeList()
        self.state = 1170
        self.match(fugue_sqlParser.EOF)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx