from antlr4.IntervalSet import IntervalSet
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import ATNState, DecisionState
def nextTokensNoContext(self, s: ATNState):
    if s.nextTokenWithinRule is not None:
        return s.nextTokenWithinRule
    s.nextTokenWithinRule = self.nextTokensInContext(s, None)
    s.nextTokenWithinRule.readonly = True
    return s.nextTokenWithinRule