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
def closure_(self, config: ATNConfig, configs: ATNConfigSet, closureBusy: set, collectPredicates: bool, fullCtx: bool, depth: int, treatEofAsEpsilon: bool):
    p = config.state
    if not p.epsilonOnlyTransitions:
        configs.add(config, self.mergeCache)
    first = True
    for t in p.transitions:
        if first:
            first = False
            if self.canDropLoopEntryEdgeInLeftRecursiveRule(config):
                continue
        continueCollecting = collectPredicates and (not isinstance(t, ActionTransition))
        c = self.getEpsilonTarget(config, t, continueCollecting, depth == 0, fullCtx, treatEofAsEpsilon)
        if c is not None:
            newDepth = depth
            if isinstance(config.state, RuleStopState):
                if self._dfa is not None and self._dfa.precedenceDfa:
                    if t.outermostPrecedenceReturn == self._dfa.atnStartState.ruleIndex:
                        c.precedenceFilterSuppressed = True
                c.reachesIntoOuterContext += 1
                if c in closureBusy:
                    continue
                closureBusy.add(c)
                configs.dipsIntoOuterContext = True
                newDepth -= 1
                if ParserATNSimulator.debug:
                    print('dips into outer ctx: ' + str(c))
            else:
                if not t.isEpsilon:
                    if c in closureBusy:
                        continue
                    closureBusy.add(c)
                if isinstance(t, RuleTransition):
                    if newDepth >= 0:
                        newDepth += 1
            self.closureCheckingStopState(c, configs, closureBusy, continueCollecting, fullCtx, newDepth, treatEofAsEpsilon)