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
def precedenceTransition(self, config: ATNConfig, pt: PrecedencePredicateTransition, collectPredicates: bool, inContext: bool, fullCtx: bool):
    if ParserATNSimulator.debug:
        print('PRED (collectPredicates=' + str(collectPredicates) + ') ' + str(pt.precedence) + '>=_p, ctx dependent=true')
        if self.parser is not None:
            print('context surrounding pred is ' + str(self.parser.getRuleInvocationStack()))
    c = None
    if collectPredicates and inContext:
        if fullCtx:
            currentPosition = self._input.index
            self._input.seek(self._startIndex)
            predSucceeds = pt.getPredicate().eval(self.parser, self._outerContext)
            self._input.seek(currentPosition)
            if predSucceeds:
                c = ATNConfig(state=pt.target, config=config)
        else:
            newSemCtx = andContext(config.semanticContext, pt.getPredicate())
            c = ATNConfig(state=pt.target, semantic=newSemCtx, config=config)
    else:
        c = ATNConfig(state=pt.target, config=config)
    if ParserATNSimulator.debug:
        print('config from pred transition=' + str(c))
    return c