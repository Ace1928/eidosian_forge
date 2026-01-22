from antlr4.IntervalSet import IntervalSet
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import ATNState, DecisionState
def defineDecisionState(self, s: DecisionState):
    self.decisionToState.append(s)
    s.decision = len(self.decisionToState) - 1
    return s.decision