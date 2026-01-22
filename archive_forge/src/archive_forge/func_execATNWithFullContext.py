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
def execATNWithFullContext(self, dfa: DFA, D: DFAState, s0: ATNConfigSet, input: TokenStream, startIndex: int, outerContext: ParserRuleContext):
    if ParserATNSimulator.debug or ParserATNSimulator.debug_list_atn_decisions:
        print('execATNWithFullContext', str(s0))
    fullCtx = True
    foundExactAmbig = False
    reach = None
    previous = s0
    input.seek(startIndex)
    t = input.LA(1)
    predictedAlt = -1
    while True:
        reach = self.computeReachSet(previous, t, fullCtx)
        if reach is None:
            e = self.noViableAlt(input, outerContext, previous, startIndex)
            input.seek(startIndex)
            alt = self.getSynValidOrSemInvalidAltThatFinishedDecisionEntryRule(previous, outerContext)
            if alt != ATN.INVALID_ALT_NUMBER:
                return alt
            else:
                raise e
        altSubSets = PredictionMode.getConflictingAltSubsets(reach)
        if ParserATNSimulator.debug:
            print('LL altSubSets=' + str(altSubSets) + ', predict=' + str(PredictionMode.getUniqueAlt(altSubSets)) + ', resolvesToJustOneViableAlt=' + str(PredictionMode.resolvesToJustOneViableAlt(altSubSets)))
        reach.uniqueAlt = self.getUniqueAlt(reach)
        if reach.uniqueAlt != ATN.INVALID_ALT_NUMBER:
            predictedAlt = reach.uniqueAlt
            break
        elif self.predictionMode is not PredictionMode.LL_EXACT_AMBIG_DETECTION:
            predictedAlt = PredictionMode.resolvesToJustOneViableAlt(altSubSets)
            if predictedAlt != ATN.INVALID_ALT_NUMBER:
                break
        elif PredictionMode.allSubsetsConflict(altSubSets) and PredictionMode.allSubsetsEqual(altSubSets):
            foundExactAmbig = True
            predictedAlt = PredictionMode.getSingleViableAlt(altSubSets)
            break
        previous = reach
        if t != Token.EOF:
            input.consume()
            t = input.LA(1)
    if reach.uniqueAlt != ATN.INVALID_ALT_NUMBER:
        self.reportContextSensitivity(dfa, predictedAlt, reach, startIndex, input.index)
        return predictedAlt
    self.reportAmbiguity(dfa, D, startIndex, input.index, foundExactAmbig, None, reach)
    return predictedAlt