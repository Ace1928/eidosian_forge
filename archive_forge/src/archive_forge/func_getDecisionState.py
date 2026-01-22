from antlr4.IntervalSet import IntervalSet
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import ATNState, DecisionState
def getDecisionState(self, decision: int):
    if len(self.decisionToState) == 0:
        return None
    else:
        return self.decisionToState[decision]