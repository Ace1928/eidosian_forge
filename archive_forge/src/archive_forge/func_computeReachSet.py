import sys
from antlr4 import DFA
from antlr4.PredictionContext import PredictionContextCache, PredictionContext, SingletonPredictionContext, \
from antlr4.BufferedTokenStream import TokenStream
from antlr4.Parser import Parser
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.Utils import str_list
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNSimulator import ATNSimulator
from antlr4.atn.ATNState import StarLoopEntryState, DecisionState, RuleStopState, ATNState
from antlr4.atn.PredictionMode import PredictionMode
from antlr4.atn.SemanticContext import SemanticContext, AND, andContext, orContext
from antlr4.atn.Transition import Transition, RuleTransition, ActionTransition, PrecedencePredicateTransition, \
from antlr4.dfa.DFAState import DFAState, PredPrediction
from antlr4.error.Errors import NoViableAltException
def computeReachSet(self, closure: ATNConfigSet, t: int, fullCtx: bool):
    if ParserATNSimulator.debug:
        print('in computeReachSet, starting closure: ' + str(closure))
    if self.mergeCache is None:
        self.mergeCache = dict()
    intermediate = ATNConfigSet(fullCtx)
    skippedStopStates = None
    for c in closure:
        if ParserATNSimulator.debug:
            print('testing ' + self.getTokenName(t) + ' at ' + str(c))
        if isinstance(c.state, RuleStopState):
            if fullCtx or t == Token.EOF:
                if skippedStopStates is None:
                    skippedStopStates = list()
                skippedStopStates.append(c)
            continue
        for trans in c.state.transitions:
            target = self.getReachableTarget(trans, t)
            if target is not None:
                intermediate.add(ATNConfig(state=target, config=c), self.mergeCache)
    reach = None
    if skippedStopStates is None and t != Token.EOF:
        if len(intermediate) == 1:
            reach = intermediate
        elif self.getUniqueAlt(intermediate) != ATN.INVALID_ALT_NUMBER:
            reach = intermediate
    if reach is None:
        reach = ATNConfigSet(fullCtx)
        closureBusy = set()
        treatEofAsEpsilon = t == Token.EOF
        for c in intermediate:
            self.closure(c, reach, closureBusy, False, fullCtx, treatEofAsEpsilon)
    if t == Token.EOF:
        reach = self.removeAllConfigsNotInRuleStopState(reach, reach is intermediate)
    if skippedStopStates is not None and (not fullCtx or not PredictionMode.hasConfigInRuleStopState(reach)):
        for c in skippedStopStates:
            reach.add(c, self.mergeCache)
    if len(reach) == 0:
        return None
    else:
        return reach