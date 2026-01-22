from antlr4.atn.ATNState import StarLoopEntryState
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import DecisionState
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import IllegalStateException
def setPrecedenceDfa(self, precedenceDfa: bool):
    if self.precedenceDfa != precedenceDfa:
        self._states = dict()
        if precedenceDfa:
            precedenceState = DFAState(configs=ATNConfigSet())
            precedenceState.edges = []
            precedenceState.isAcceptState = False
            precedenceState.requiresFullContext = False
            self.s0 = precedenceState
        else:
            self.s0 = None
        self.precedenceDfa = precedenceDfa