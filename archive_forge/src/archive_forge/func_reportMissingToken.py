import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def reportMissingToken(self, recognizer: Parser):
    if self.inErrorRecoveryMode(recognizer):
        return
    self.beginErrorCondition(recognizer)
    t = recognizer.getCurrentToken()
    expecting = self.getExpectedTokens(recognizer)
    msg = 'missing ' + expecting.toString(recognizer.literalNames, recognizer.symbolicNames) + ' at ' + self.getTokenErrorDisplay(t)
    recognizer.notifyErrorListeners(msg, t, None)