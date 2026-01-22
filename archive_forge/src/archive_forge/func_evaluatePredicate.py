from antlr4.PredictionContext import PredictionContextCache, SingletonPredictionContext, PredictionContext
from antlr4.InputStream import InputStream
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import LexerATNConfig
from antlr4.atn.ATNSimulator import ATNSimulator
from antlr4.atn.ATNConfigSet import ATNConfigSet, OrderedATNConfigSet
from antlr4.atn.ATNState import RuleStopState, ATNState
from antlr4.atn.LexerActionExecutor import LexerActionExecutor
from antlr4.atn.Transition import Transition
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import LexerNoViableAltException, UnsupportedOperationException
def evaluatePredicate(self, input: InputStream, ruleIndex: int, predIndex: int, speculative: bool):
    if self.recog is None:
        return True
    if not speculative:
        return self.recog.sempred(None, ruleIndex, predIndex)
    savedcolumn = self.column
    savedLine = self.line
    index = input.index
    marker = input.mark()
    try:
        self.consume(input)
        return self.recog.sempred(None, ruleIndex, predIndex)
    finally:
        self.column = savedcolumn
        self.line = savedLine
        input.seek(index)
        input.release(marker)