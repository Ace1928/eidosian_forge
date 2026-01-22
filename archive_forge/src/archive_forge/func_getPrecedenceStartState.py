from antlr4.atn.ATNState import StarLoopEntryState
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import DecisionState
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import IllegalStateException
def getPrecedenceStartState(self, precedence: int):
    if not self.precedenceDfa:
        raise IllegalStateException('Only precedence DFAs may contain a precedence start state.')
    if precedence < 0 or precedence >= len(self.s0.edges):
        return None
    return self.s0.edges[precedence]