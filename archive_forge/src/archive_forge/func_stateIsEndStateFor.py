from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def stateIsEndStateFor(self, state: ATNState, idx: int):
    if state.ruleIndex != idx:
        return None
    if not isinstance(state, StarLoopEntryState):
        return None
    maybeLoopEndState = state.transitions[len(state.transitions) - 1].target
    if not isinstance(maybeLoopEndState, LoopEndState):
        return None
    if maybeLoopEndState.epsilonOnlyTransitions and isinstance(maybeLoopEndState.transitions[0].target, RuleStopState):
        return state
    else:
        return None