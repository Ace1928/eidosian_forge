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
def execATN(self, input: InputStream, ds0: DFAState):
    if LexerATNSimulator.debug:
        print('start state closure=' + str(ds0.configs))
    if ds0.isAcceptState:
        self.captureSimState(self.prevAccept, input, ds0)
    t = input.LA(1)
    s = ds0
    while True:
        if LexerATNSimulator.debug:
            print('execATN loop starting closure:', str(s.configs))
        target = self.getExistingTargetState(s, t)
        if target is None:
            target = self.computeTargetState(input, s, t)
        if target == self.ERROR:
            break
        if t != Token.EOF:
            self.consume(input)
        if target.isAcceptState:
            self.captureSimState(self.prevAccept, input, target)
            if t == Token.EOF:
                break
        t = input.LA(1)
        s = target
    return self.failOrAccept(self.prevAccept, input, s.configs, t)