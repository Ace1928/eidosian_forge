from antlr4 import *
from io import StringIO
import sys
def fugueLanguage(self):
    localctx = fugue_sqlParser.FugueLanguageContext(self, self._ctx, self.state)
    self.enterRule(localctx, 0, self.RULE_fugueLanguage)
    self._la = 0
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 431
        self._errHandler.sync(self)
        _la = self._input.LA(1)
        while True:
            self.state = 430
            self.fugueSingleTask()
            self.state = 433
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if not (_la & ~63 == 0 and 1 << _la & -269793669747965952 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & 18014398509481983 != 0) or (_la - 324 & ~63 == 0 and 1 << _la - 324 & 98305 != 0)):
                break
        self.state = 435
        self.match(fugue_sqlParser.EOF)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx