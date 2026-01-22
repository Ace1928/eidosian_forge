from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def markPrecedenceDecisions(self, atn: ATN):
    for state in atn.states:
        if not isinstance(state, StarLoopEntryState):
            continue
        if atn.ruleToStartState[state.ruleIndex].isPrecedenceRule:
            maybeLoopEndState = state.transitions[len(state.transitions) - 1].target
            if isinstance(maybeLoopEndState, LoopEndState):
                if maybeLoopEndState.epsilonOnlyTransitions and isinstance(maybeLoopEndState.transitions[0].target, RuleStopState):
                    state.isPrecedenceDecision = True