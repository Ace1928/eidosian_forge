from antlr4.IntervalSet import IntervalSet
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import ATNState, DecisionState
def nextTokensInContext(self, s: ATNState, ctx: RuleContext):
    from antlr4.LL1Analyzer import LL1Analyzer
    anal = LL1Analyzer(self)
    return anal.LOOK(s, ctx=ctx)