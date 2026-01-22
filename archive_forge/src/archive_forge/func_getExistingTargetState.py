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
def getExistingTargetState(self, s: DFAState, t: int):
    if s.edges is None or t < self.MIN_DFA_EDGE or t > self.MAX_DFA_EDGE:
        return None
    target = s.edges[t - self.MIN_DFA_EDGE]
    if LexerATNSimulator.debug and target is not None:
        print('reuse state', str(s.stateNumber), 'edge to', str(target.stateNumber))
    return target