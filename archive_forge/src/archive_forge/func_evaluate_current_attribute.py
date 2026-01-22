import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
def evaluate_current_attribute(cursor_offset, line, namespace=None):
    """Safely evaluates the expression having an attributed accessed"""
    obj = evaluate_current_expression(cursor_offset, line, namespace)
    attr = line_properties.current_expression_attribute(cursor_offset, line)
    if attr is None:
        raise EvaluationError('No attribute found to look up')
    try:
        return getattr(obj, attr.word)
    except AttributeError:
        raise EvaluationError(f"can't lookup attribute {attr.word} on {obj!r}")