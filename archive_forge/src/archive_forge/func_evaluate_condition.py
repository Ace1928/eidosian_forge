import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict
def evaluate_condition(condition, state, tools):
    if len(condition.ops) > 1:
        raise InterpretorError('Cannot evaluate conditions with multiple operators')
    left = evaluate_ast(condition.left, state, tools)
    comparator = condition.ops[0]
    right = evaluate_ast(condition.comparators[0], state, tools)
    if isinstance(comparator, ast.Eq):
        return left == right
    elif isinstance(comparator, ast.NotEq):
        return left != right
    elif isinstance(comparator, ast.Lt):
        return left < right
    elif isinstance(comparator, ast.LtE):
        return left <= right
    elif isinstance(comparator, ast.Gt):
        return left > right
    elif isinstance(comparator, ast.GtE):
        return left >= right
    elif isinstance(comparator, ast.Is):
        return left is right
    elif isinstance(comparator, ast.IsNot):
        return left is not right
    elif isinstance(comparator, ast.In):
        return left in right
    elif isinstance(comparator, ast.NotIn):
        return left not in right
    else:
        raise InterpretorError(f'Operator not supported: {comparator}')