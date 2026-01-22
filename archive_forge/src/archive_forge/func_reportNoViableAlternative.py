import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def reportNoViableAlternative(self, recognizer: Parser, e: NoViableAltException):
    tokens = recognizer.getTokenStream()
    if tokens is not None:
        if e.startToken.type == Token.EOF:
            input = '<EOF>'
        else:
            input = tokens.getText(e.startToken, e.offendingToken)
    else:
        input = '<unknown input>'
    msg = 'no viable alternative at input ' + self.escapeWSAndQuote(input)
    recognizer.notifyErrorListeners(msg, e.offendingToken, e)