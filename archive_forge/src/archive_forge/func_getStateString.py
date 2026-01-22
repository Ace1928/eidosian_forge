from io import StringIO
from antlr4 import DFA
from antlr4.Utils import str_list
from antlr4.dfa.DFAState import DFAState
def getStateString(self, s: DFAState):
    n = s.stateNumber
    baseStateStr = (':' if s.isAcceptState else '') + 's' + str(n) + ('^' if s.requiresFullContext else '')
    if s.isAcceptState:
        if s.predicates is not None:
            return baseStateStr + '=>' + str_list(s.predicates)
        else:
            return baseStateStr + '=>' + str(s.prediction)
    else:
        return baseStateStr