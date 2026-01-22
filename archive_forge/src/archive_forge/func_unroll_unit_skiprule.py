from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def unroll_unit_skiprule(lhs, orig_rhs, skipped_rules, children, weight, alias):
    if not skipped_rules:
        return RuleNode(Rule(lhs, orig_rhs, weight=weight, alias=alias), children, weight=weight)
    else:
        weight = weight - skipped_rules[0].weight
        return RuleNode(Rule(lhs, [skipped_rules[0].lhs], weight=weight, alias=alias), [unroll_unit_skiprule(skipped_rules[0].lhs, orig_rhs, skipped_rules[1:], children, skipped_rules[0].weight, skipped_rules[0].alias)], weight=weight)