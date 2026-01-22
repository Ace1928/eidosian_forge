from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def build_unit_skiprule(unit_rule, target_rule):
    skipped_rules = []
    if isinstance(unit_rule, UnitSkipRule):
        skipped_rules += unit_rule.skipped_rules
    skipped_rules.append(target_rule)
    if isinstance(target_rule, UnitSkipRule):
        skipped_rules += target_rule.skipped_rules
    return UnitSkipRule(unit_rule.lhs, target_rule.rhs, skipped_rules, weight=unit_rule.weight + target_rule.weight, alias=unit_rule.alias)