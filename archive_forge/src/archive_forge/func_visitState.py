from antlr4.dfa.DFA import DFA
from antlr4.BufferedTokenStream import TokenStream
from antlr4.Lexer import Lexer
from antlr4.Parser import Parser
from antlr4.ParserRuleContext import InterpreterRuleContext, ParserRuleContext
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import StarLoopEntryState, ATNState, LoopEndState
from antlr4.atn.ParserATNSimulator import ParserATNSimulator
from antlr4.PredictionContext import PredictionContextCache
from antlr4.atn.Transition import Transition
from antlr4.error.Errors import RecognitionException, UnsupportedOperationException, FailedPredicateException
def visitState(self, p: ATNState):
    edge = 0
    if len(p.transitions) > 1:
        self._errHandler.sync(self)
        edge = self._interp.adaptivePredict(self._input, p.decision, self._ctx)
    else:
        edge = 1
    transition = p.transitions[edge - 1]
    tt = transition.serializationType
    if tt == Transition.EPSILON:
        if self.pushRecursionContextStates[p.stateNumber] and (not isinstance(transition.target, LoopEndState)):
            t = self._parentContextStack[-1]
            ctx = InterpreterRuleContext(t[0], t[1], self._ctx.ruleIndex)
            self.pushNewRecursionContext(ctx, self.atn.ruleToStartState[p.ruleIndex].stateNumber, self._ctx.ruleIndex)
    elif tt == Transition.ATOM:
        self.match(transition.label)
    elif tt in [Transition.RANGE, Transition.SET, Transition.NOT_SET]:
        if not transition.matches(self._input.LA(1), Token.MIN_USER_TOKEN_TYPE, Lexer.MAX_CHAR_VALUE):
            self._errHandler.recoverInline(self)
        self.matchWildcard()
    elif tt == Transition.WILDCARD:
        self.matchWildcard()
    elif tt == Transition.RULE:
        ruleStartState = transition.target
        ruleIndex = ruleStartState.ruleIndex
        ctx = InterpreterRuleContext(self._ctx, p.stateNumber, ruleIndex)
        if ruleStartState.isPrecedenceRule:
            self.enterRecursionRule(ctx, ruleStartState.stateNumber, ruleIndex, transition.precedence)
        else:
            self.enterRule(ctx, transition.target.stateNumber, ruleIndex)
    elif tt == Transition.PREDICATE:
        if not self.sempred(self._ctx, transition.ruleIndex, transition.predIndex):
            raise FailedPredicateException(self)
    elif tt == Transition.ACTION:
        self.action(self._ctx, transition.ruleIndex, transition.actionIndex)
    elif tt == Transition.PRECEDENCE:
        if not self.precpred(self._ctx, transition.precedence):
            msg = 'precpred(_ctx, ' + str(transition.precedence) + ')'
            raise FailedPredicateException(self, msg)
    else:
        raise UnsupportedOperationException('Unrecognized ATN transition type.')
    self.state = transition.target.stateNumber