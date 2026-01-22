import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def singleTokenInsertion(self, recognizer: Parser):
    currentSymbolType = recognizer.getTokenStream().LA(1)
    atn = recognizer._interp.atn
    currentState = atn.states[recognizer.state]
    next = currentState.transitions[0].target
    expectingAtLL2 = atn.nextTokens(next, recognizer._ctx)
    if currentSymbolType in expectingAtLL2:
        self.reportMissingToken(recognizer)
        return True
    else:
        return False