from inspect import signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.decorators import undoc
def eval_node(node: Union[ast.AST, None], context: EvaluationContext):
    """Evaluate AST node in provided context.

    Applies evaluation restrictions defined in the context. Currently does not support evaluation of functions with keyword arguments.

    Does not evaluate actions that always have side effects:

    - class definitions (``class sth: ...``)
    - function definitions (``def sth: ...``)
    - variable assignments (``x = 1``)
    - augmented assignments (``x += 1``)
    - deletions (``del x``)

    Does not evaluate operations which do not return values:

    - assertions (``assert x``)
    - pass (``pass``)
    - imports (``import x``)
    - control flow:

        - conditionals (``if x:``) except for ternary IfExp (``a if x else b``)
        - loops (``for`` and ``while``)
        - exception handling

    The purpose of this function is to guard against unwanted side-effects;
    it does not give guarantees on protection from malicious code execution.
    """
    policy = EVALUATION_POLICIES[context.evaluation]
    if node is None:
        return None
    if isinstance(node, ast.Expression):
        return eval_node(node.body, context)
    if isinstance(node, ast.BinOp):
        left = eval_node(node.left, context)
        right = eval_node(node.right, context)
        dunders = _find_dunder(node.op, BINARY_OP_DUNDERS)
        if dunders:
            if policy.can_operate(dunders, left, right):
                return getattr(left, dunders[0])(right)
            else:
                raise GuardRejection(f'Operation (`{dunders}`) for', type(left), f'not allowed in {context.evaluation} mode')
    if isinstance(node, ast.Compare):
        left = eval_node(node.left, context)
        all_true = True
        negate = False
        for op, right in zip(node.ops, node.comparators):
            right = eval_node(right, context)
            dunder = None
            dunders = _find_dunder(op, COMP_OP_DUNDERS)
            if not dunders:
                if isinstance(op, ast.NotIn):
                    dunders = COMP_OP_DUNDERS[ast.In]
                    negate = True
                if isinstance(op, ast.Is):
                    dunder = 'is_'
                if isinstance(op, ast.IsNot):
                    dunder = 'is_'
                    negate = True
            if not dunder and dunders:
                dunder = dunders[0]
            if dunder:
                a, b = (right, left) if dunder == '__contains__' else (left, right)
                if dunder == 'is_' or (dunders and policy.can_operate(dunders, a, b)):
                    result = getattr(operator, dunder)(a, b)
                    if negate:
                        result = not result
                    if not result:
                        all_true = False
                    left = right
                else:
                    raise GuardRejection(f'Comparison (`{dunder}`) for', type(left), f'not allowed in {context.evaluation} mode')
            else:
                raise ValueError(f'Comparison `{dunder}` not supported')
        return all_true
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple((eval_node(e, context) for e in node.elts))
    if isinstance(node, ast.List):
        return [eval_node(e, context) for e in node.elts]
    if isinstance(node, ast.Set):
        return {eval_node(e, context) for e in node.elts}
    if isinstance(node, ast.Dict):
        return dict(zip([eval_node(k, context) for k in node.keys], [eval_node(v, context) for v in node.values]))
    if isinstance(node, ast.Slice):
        return slice(eval_node(node.lower, context), eval_node(node.upper, context), eval_node(node.step, context))
    if isinstance(node, ast.UnaryOp):
        value = eval_node(node.operand, context)
        dunders = _find_dunder(node.op, UNARY_OP_DUNDERS)
        if dunders:
            if policy.can_operate(dunders, value):
                return getattr(value, dunders[0])()
            else:
                raise GuardRejection(f'Operation (`{dunders}`) for', type(value), f'not allowed in {context.evaluation} mode')
    if isinstance(node, ast.Subscript):
        value = eval_node(node.value, context)
        slice_ = eval_node(node.slice, context)
        if policy.can_get_item(value, slice_):
            return value[slice_]
        raise GuardRejection('Subscript access (`__getitem__`) for', type(value), f' not allowed in {context.evaluation} mode')
    if isinstance(node, ast.Name):
        if policy.allow_locals_access and node.id in context.locals:
            return context.locals[node.id]
        if policy.allow_globals_access and node.id in context.globals:
            return context.globals[node.id]
        if policy.allow_builtins_access and hasattr(builtins, node.id):
            return getattr(builtins, node.id)
        if not policy.allow_globals_access and (not policy.allow_locals_access):
            raise GuardRejection(f'Namespace access not allowed in {context.evaluation} mode')
        else:
            raise NameError(f'{node.id} not found in locals, globals, nor builtins')
    if isinstance(node, ast.Attribute):
        value = eval_node(node.value, context)
        if policy.can_get_attr(value, node.attr):
            return getattr(value, node.attr)
        raise GuardRejection('Attribute access (`__getattr__`) for', type(value), f'not allowed in {context.evaluation} mode')
    if isinstance(node, ast.IfExp):
        test = eval_node(node.test, context)
        if test:
            return eval_node(node.body, context)
        else:
            return eval_node(node.orelse, context)
    if isinstance(node, ast.Call):
        func = eval_node(node.func, context)
        if policy.can_call(func) and (not node.keywords):
            args = [eval_node(arg, context) for arg in node.args]
            return func(*args)
        try:
            sig = signature(func)
        except ValueError:
            sig = UNKNOWN_SIGNATURE
        not_empty = sig.return_annotation is not Signature.empty
        not_stringized = not isinstance(sig.return_annotation, str)
        if not_empty and not_stringized:
            duck = Duck()
            if policy.can_call(sig.return_annotation) and (not node.keywords):
                args = [eval_node(arg, context) for arg in node.args]
                return sig.return_annotation(*args)
            try:
                duck.__class__ = sig.return_annotation
                return duck
            except TypeError:
                pass
        raise GuardRejection('Call for', func, f'not allowed in {context.evaluation} mode')
    raise ValueError('Unhandled node', ast.dump(node))