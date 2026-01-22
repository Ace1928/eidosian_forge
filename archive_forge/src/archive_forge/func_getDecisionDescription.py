from io import StringIO
from antlr4 import Parser, DFA
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.error.ErrorListener import ErrorListener
def getDecisionDescription(self, recognizer: Parser, dfa: DFA):
    decision = dfa.decision
    ruleIndex = dfa.atnStartState.ruleIndex
    ruleNames = recognizer.ruleNames
    if ruleIndex < 0 or ruleIndex >= len(ruleNames):
        return str(decision)
    ruleName = ruleNames[ruleIndex]
    if ruleName is None or len(ruleName) == 0:
        return str(decision)
    return str(decision) + ' (' + ruleName + ')'