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
def guarded_eval(code: str, context: EvaluationContext):
    """Evaluate provided code in the evaluation context.

    If evaluation policy given by context is set to ``forbidden``
    no evaluation will be performed; if it is set to ``dangerous``
    standard :func:`eval` will be used; finally, for any other,
    policy :func:`eval_node` will be called on parsed AST.
    """
    locals_ = context.locals
    if context.evaluation == 'forbidden':
        raise GuardRejection('Forbidden mode')
    if context.in_subscript:
        if not code:
            return tuple()
        locals_ = locals_.copy()
        locals_[SUBSCRIPT_MARKER] = IDENTITY_SUBSCRIPT
        code = SUBSCRIPT_MARKER + '[' + code + ']'
        context = EvaluationContext(**{**context._asdict(), **{'locals': locals_}})
    if context.evaluation == 'dangerous':
        return eval(code, context.globals, context.locals)
    expression = ast.parse(code, mode='eval')
    return eval_node(expression, context)