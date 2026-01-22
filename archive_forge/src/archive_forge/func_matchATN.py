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
def matchATN(self, input: InputStream):
    startState = self.atn.modeToStartState[self.mode]
    if LexerATNSimulator.debug:
        print('matchATN mode ' + str(self.mode) + ' start: ' + str(startState))
    old_mode = self.mode
    s0_closure = self.computeStartState(input, startState)
    suppressEdge = s0_closure.hasSemanticContext
    s0_closure.hasSemanticContext = False
    next = self.addDFAState(s0_closure)
    if not suppressEdge:
        self.decisionToDFA[self.mode].s0 = next
    predict = self.execATN(input, next)
    if LexerATNSimulator.debug:
        print('DFA after matchATN: ' + str(self.decisionToDFA[old_mode].toLexerString()))
    return predict